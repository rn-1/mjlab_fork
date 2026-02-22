"""
CollectionEnv: a RslRlVecEnvWrapper subclass that hooks Interlatent trajectory
and activation collection into a mjlab environment.

Two usage modes (wandb-style):

    Mode 1 – standalone collection (drives its own rollout loop):

        db = LatentDB("sqlite:///run.db")
        env = ManagerBasedRlEnv(cfg=env_cfg)
        col_env = CollectionEnv(env, db=db, hook_layers=["mlp"])
        runner = MjlabOnPolicyRunner(col_env, asdict(agent_cfg), device=device)
        runner.load(checkpoint_path, load_cfg={"actor": True}, strict=True)

        run_info = col_env.collect(
            actor_model=runner.alg.actor,
            steps=2000,
            tags={"task": "G1-velocity"},
        )

    Mode 2 – passive collection (hooks into an existing training/eval loop):

        db = LatentDB("sqlite:///run.db")
        env = ManagerBasedRlEnv(cfg=env_cfg)
        col_env = CollectionEnv(env, db=db, hook_layers=["mlp"])
        runner = MjlabOnPolicyRunner(col_env, asdict(agent_cfg), device=device)

        with col_env.collecting(runner.alg.actor):
            runner.learn(max_iterations=500)

Notes
-----
- Both modes track env index 0 only. For multi-env collection run several
  independent processes with separate .db files and merge them.
- Mode 1 requires num_envs=1 (or at most env 0 is observed). Mode 2 works
  with any num_envs but still records env 0 for trajectory context.
- The actor_model can be pre-registered via attach() instead of passing it
  to collect() / collecting() every time.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper


# Default number of pending step-context entries to accumulate before
# flushing to the database in Mode 2.
_DEFAULT_CONTEXT_FLUSH_EVERY = 256


class CollectionEnv(RslRlVecEnvWrapper):
    """RslRlVecEnvWrapper that hooks Interlatent trajectory and activation collection.

    Drop-in replacement for RslRlVecEnvWrapper.  All runner / PPO machinery sees
    the standard VecEnv interface; collection is layered on top without changing
    the training contract.

    Parameters
    ----------
    env:
        The unwrapped ManagerBasedRlEnv.
    db:
        An open LatentDB instance.  The caller owns its lifecycle (close / flush).
    hook_layers:
        Dotted layer names relative to the actor MLPModel to hook, e.g.
        ``["mlp", "mlp.0"]``.  Run ``--list-layers`` in collect.py to enumerate.
    actor_obs_key:
        Key used to extract the actor observation from the TensorDict returned
        by the vec-env (default: ``"actor"``).
    clip_actions:
        Forwarded unchanged to RslRlVecEnvWrapper.
    context_flush_every:
        In Mode 2, how many pending step contexts to buffer before writing
        them to the database.
    """

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        *,
        db,  # LatentDB – avoid hard import so the class loads without interlatent
        hook_layers: Sequence[str] = ("mlp",),
        actor_obs_key: str = "actor",
        clip_actions: float | None = None,
        context_flush_every: int = _DEFAULT_CONTEXT_FLUSH_EVERY,
    ) -> None:
        super().__init__(env, clip_actions=clip_actions)

        self._db = db
        self._hook_layers = list(hook_layers)
        self._actor_obs_key = actor_obs_key
        self._context_flush_every = context_flush_every

        # Set by attach() or passed directly to collect() / collecting()
        self._actor_model: torch.nn.Module | None = None

        # ── Mode 2 state ────────────────────────────────────────────────
        # Mutated in-place so the StepAlignedHookCtx closure always reads
        # the latest value.
        self._step_ctx: Dict[str, Any] = {}

        self._collecting: bool = False
        self._active_run_id: str = ""
        self._global_step: int = 0
        self._episode_id: int = 0
        self._episode_step: int = 0
        self._pending_contexts: Dict[int, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # wandb.watch() equivalent
    # ------------------------------------------------------------------

    def attach(self, actor_model: torch.nn.Module) -> None:
        """Pre-register the actor model so collect() / collecting() need no argument.

        Equivalent to ``wandb.watch(model)`` – call once after runner is built.

        Example
        -------
        >>> col_env.attach(runner.alg.actor)
        >>> col_env.collect(steps=2000)
        """
        self._actor_model = actor_model

    # ------------------------------------------------------------------
    # Mode 1 – standalone collection
    # ------------------------------------------------------------------

    def collect(
        self,
        *,
        steps: int,
        actor_model: torch.nn.Module | None = None,
        policy_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        context_fn: Optional[Callable[..., Dict[str, Any]]] = None,
        run_id: str | None = None,
        tags: Dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Drive a standalone rollout loop, collecting trajectories and activations.

        Wraps this env in a single-env gym shim and delegates to
        ``TrajectoryCollector.run()``.  The actor's forward hooks fire
        automatically during each policy call.

        Parameters
        ----------
        steps:
            Total environment steps to collect.
        actor_model:
            Actor nn.Module to hook.  Falls back to the model set by attach().
        policy_fn:
            ``obs_np -> action_np`` callable.  If *None*, a default wrapper
            around *actor_model* is built automatically.
        context_fn:
            Optional ``(**step_kwargs) -> dict`` called each step to supply
            extra context written alongside activations.
        run_id:
            Override the auto-generated UUID for this run.
        tags:
            Arbitrary key-value metadata stored with the run.
        **kwargs:
            Forwarded to ``TrajectoryCollector.run()`` (e.g. ``max_episodes``).

        Returns
        -------
        RunInfo
            Interlatent RunInfo object with ``run_id`` and metadata.
        """
        from interlatent.collectors.trajectory_collector import TrajectoryCollector

        model = self._resolve_model(actor_model)
        if policy_fn is None:
            policy_fn = _make_rsl_policy_fn(model, self._actor_obs_key, self.device)

        shim = _GymShim(self, actor_obs_key=self._actor_obs_key)
        collector = TrajectoryCollector(self._db, hook_layers=self._hook_layers)

        return collector.run(
            model=model,
            env=shim,
            steps=steps,
            policy_fn=policy_fn,
            context_fn=context_fn,
            scenario_id=run_id,
            tags=tags or {},
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Mode 2 – passive collection (hooks into an external loop)
    # ------------------------------------------------------------------

    @contextmanager
    def collecting(
        self,
        actor_model: torch.nn.Module | None = None,
        *,
        run_id: str | None = None,
        tags: Dict[str, Any] | None = None,
    ):
        """Context manager that hooks collection into an existing loop.

        Attaches ``StepAlignedHookCtx`` to the actor and intercepts every
        ``env.step()`` call to maintain a shared step-context dict.  The
        hook reads this dict on each forward pass, so activations are
        automatically correlated with the correct global step, episode id,
        and intra-episode timestep.

        Usage
        -----
        >>> with col_env.collecting(runner.alg.actor) as run_id:
        ...     runner.learn(max_iterations=500)
        >>> print("run:", run_id)

        Parameters
        ----------
        actor_model:
            Actor nn.Module to hook.  Falls back to the model set by attach().
        run_id:
            Override the auto-generated UUID.
        tags:
            Arbitrary key-value metadata (currently stored in step contexts).

        Yields
        ------
        str
            The run UUID (useful for downstream pipeline calls).

        Notes
        -----
        Step context alignment: the hook fires inside ``alg.act()`` BEFORE
        ``env.step()`` is called.  The step counter in ``_step_ctx`` is
        initialised to 0 and incremented AT THE END of each ``env.step()``
        call.  This ensures the hook at iteration N reads step=N.
        """
        from interlatent.hooks import StepAlignedHookCtx

        model = self._resolve_model(actor_model)
        effective_run_id = run_id or uuid.uuid4().hex

        # Reset all per-run state
        self._active_run_id = effective_run_id
        self._global_step = 0
        self._episode_id = 0
        self._episode_step = 0
        self._pending_contexts.clear()

        # Prime the context dict so the first hook call at step 0 is correct.
        self._step_ctx.clear()
        self._step_ctx.update({"step": 0, "episode_id": 0, "t": 0})
        if tags:
            self._step_ctx.update(tags)

        hook_ctx = StepAlignedHookCtx(
            model,
            layers=self._hook_layers,
            db=self._db,
            run_id=effective_run_id,
            context_supplier=lambda: self._step_ctx,
            step_key="step",
        )

        self._collecting = True
        try:
            with hook_ctx:
                yield effective_run_id
        finally:
            self._collecting = False
            self._flush_pending_contexts()
            self._db.flush()

    # ------------------------------------------------------------------
    # Overrides – intercept step/reset for Mode 2 context tracking
    # ------------------------------------------------------------------

    def step(
        self, actions: torch.Tensor
    ) -> tuple:
        # Capture the action BEFORE calling super so it's available in context.
        action_snapshot: list | None = None
        if self._collecting:
            action_snapshot = actions[0].detach().cpu().tolist()

        obs_td, rew, dones, extras = super().step(actions)

        if self._collecting:
            gs = self._global_step
            time_outs = extras.get(
                "time_outs", torch.zeros(self.num_envs, device=self.device)
            )

            self._pending_contexts[gs] = {
                "step": gs,
                "episode_id": self._episode_id,
                "t": self._episode_step,
                "action": action_snapshot,
                "reward": float(rew[0].item()),
                "done": bool(dones[0].item()),
                "truncated": bool(time_outs[0].item()),
            }

            if len(self._pending_contexts) >= self._context_flush_every:
                self._flush_pending_contexts()

            # Advance counters; update step_ctx for the NEXT alg.act() call.
            self._episode_step += 1
            if bool(dones[0].item()):
                self._episode_id += 1
                self._episode_step = 0

            self._global_step += 1
            self._step_ctx.update(
                {
                    "step": self._global_step,
                    "episode_id": self._episode_id,
                    "t": self._episode_step,
                }
            )

        return obs_td, rew, dones, extras

    def reset(self) -> tuple:
        result = super().reset()
        if self._collecting:
            self._episode_step = 0
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_model(self, actor_model: torch.nn.Module | None) -> torch.nn.Module:
        model = actor_model or self._actor_model
        if model is None:
            raise RuntimeError(
                "No actor model available.  Either pass actor_model= or call "
                "col_env.attach(runner.alg.actor) after building the runner."
            )
        return model

    def _flush_pending_contexts(self) -> None:
        if self._pending_contexts:
            self._db.update_step_contexts(
                run_id=self._active_run_id,
                contexts=self._pending_contexts,
            )
            self._pending_contexts.clear()


# ---------------------------------------------------------------------------
# Internal gym shim (Mode 1 helper)
# ---------------------------------------------------------------------------


class _GymShim:
    """Adapt a batched CollectionEnv to the single-env gym interface that
    TrajectoryCollector.run() expects.

    Squeezes the batch dimension (env index 0) and converts tensors to numpy.
    Mirrors the role of _MjlabGymShim in collect.py but is tied to
    CollectionEnv rather than requiring an external shim.
    """

    def __init__(self, vec_env: CollectionEnv, *, actor_obs_key: str = "actor") -> None:
        self._env = vec_env
        self._actor_obs_key = actor_obs_key
        # Expose action_space so TrajectoryCollector can inspect it if needed.
        self.action_space = vec_env.unwrapped.action_space
        self.spec = None

    def _obs_to_array(self, obs_td) -> np.ndarray:
        return obs_td[self._actor_obs_key].squeeze(0).cpu().numpy()

    def reset(self) -> tuple[np.ndarray, dict]:
        obs_td, extras = self._env.reset()
        return self._obs_to_array(obs_td), extras

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_t = (
            torch.as_tensor(action, dtype=torch.float32)
            .unsqueeze(0)
            .to(self._env.device)
        )
        obs_td, rew, dones, extras = self._env.step(action_t)
        obs = self._obs_to_array(obs_td)
        reward = float(rew[0].item())
        done = bool(dones[0].item())
        truncated = bool(
            extras.get("time_outs", torch.zeros(1))[0].item()
        )
        return obs, reward, done, truncated, extras

    def render(self) -> None:
        return None

    def close(self) -> None:
        self._env.close()


# ---------------------------------------------------------------------------
# Policy factory (Mode 1 default)
# ---------------------------------------------------------------------------


def _make_rsl_policy_fn(
    actor_model: torch.nn.Module,
    actor_obs_key: str,
    device: torch.device,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a numpy-obs → numpy-action policy function from an RSL-RL actor.

    Mirrors the policy_fn built inside collect.py's main().
    """
    from tensordict import TensorDict

    def policy_fn(obs_np: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        obs_td = TensorDict({actor_obs_key: obs_t}, batch_size=[1])
        with torch.no_grad():
            action_t = actor_model(obs_td)
        return action_t.squeeze(0).cpu().numpy()

    return policy_fn
