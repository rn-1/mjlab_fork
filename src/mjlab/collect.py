"""Collect mjlab policy activations into a LatentDB for Interlatent analysis.

Runs rollouts with a trained mjlab checkpoint and uses TrajectoryCollector to
capture layer activations and per-step context into a SQLite database, ready
for the Interlatent pipeline (SAE training, dashboard export).

Usage
-----
    uv run python collect_for_db.py \\
        --task Mjlab-Velocity-Flat-Unitree-G1 \\
        --checkpoint logs/rsl_rl/g1_velocity/<run>/model_1000.pt \\
        --db g1_run.db \\
        --steps 2000 \\
        --layer mlp

List available layer names in the actor network:

    uv run python collect_for_db.py \\
        --task Mjlab-Velocity-Flat-Unitree-G1 \\
        --checkpoint path/to/model.pt \\
        --list-layers

Then process with the Interlatent pipeline:

    cd ../Interlatent-Robotics/pipeline
    python mvp.py process \\
        --db ../../mjlab_fork/g1_run.db \\
        --run-id <run_id from above> \\
        --layer mlp \\
        --k 32

Notes
-----
- Uses num_envs=1 so the gym-shim adapter works simply. For faster collection
  consider running multiple independent collect_for_db.py processes each with a
  different --scenario-id and merging the resulting .db files.
- The hook fires on runner.alg.actor (the MLPModel), not the full runner. Pass
  dotted layer names relative to that module (e.g. "mlp", "mlp.0").
- For tracking tasks you must also pass --motion-file.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Gym-compatible shim around RslRlVecEnvWrapper (num_envs=1)
# ---------------------------------------------------------------------------

class _MjlabGymShim:
    """Wraps RslRlVecEnvWrapper to look like a single-env gymnasium interface.

    TrajectoryCollector expects:
      reset() -> (obs_array, info_dict)
      step(action_array) -> (obs_array, float, bool, bool, info_dict)
      render() -> np.ndarray | None

    RslRlVecEnvWrapper returns batched tensors (num_envs=1 here), so we just
    squeeze the batch dimension and convert to numpy.
    """

    def __init__(self, vec_env, actor_obs_key: str = "actor"):
        self._env = vec_env
        self._actor_obs_key = actor_obs_key
        # Expose action_space so TrajectoryCollector can inspect it if needed.
        self.action_space = vec_env.unwrapped.action_space
        self.spec = None  # no gym spec; env_name taken from class name

    def _obs_to_array(self, obs_td):
        """Extract actor obs from TensorDict and return as 1-D numpy array."""
        arr = obs_td[self._actor_obs_key]  # (1, obs_dim)
        return arr.squeeze(0).cpu().numpy()

    def reset(self):
        obs_td, extras = self._env.reset()
        return self._obs_to_array(obs_td), extras

    def step(self, action):
        action_t = torch.as_tensor(action, dtype=torch.float32).unsqueeze(0).to(
            self._env.device
        )
        obs_td, reward_t, dones_t, extras = self._env.step(action_t)
        obs = self._obs_to_array(obs_td)
        reward = float(reward_t[0].item())
        done = bool(dones_t[0].item())
        truncated = bool(extras.get("time_outs", torch.zeros(1))[0].item())
        return obs, reward, done, truncated, extras

    def render(self):
        return None

    def close(self):
        self._env.close()


# ---------------------------------------------------------------------------
# Observation context builder
# ---------------------------------------------------------------------------

def _build_context_fn(obs_names: list[str] | None):
    """Return a context_fn that records named observations each step."""
    if not obs_names:
        def _ctx(next_obs, **_):
            if next_obs is None:
                return {}
            return {"observations": {f"obs_{i}": float(v) for i, v in enumerate(next_obs)}}
        return _ctx

    def _ctx(next_obs, **_):
        if next_obs is None:
            return {}
        n = min(len(obs_names), len(next_obs))
        return {"observations": {obs_names[i]: float(next_obs[i]) for i in range(n)}}
    return _ctx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect mjlab policy activations into a LatentDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", required=True,
                        help="Registered task ID, e.g. Mjlab-Velocity-Flat-Unitree-G1")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model_*.pt checkpoint file")
    parser.add_argument("--db", default="mjlab_run.db",
                        help="Output SQLite database path")
    parser.add_argument("--steps", type=int, default=2000,
                        help="Total environment steps to collect")
    parser.add_argument("--layer", default="mlp",
                        help="Dotted layer name on the actor MLPModel to hook "
                             "(e.g. 'mlp', 'mlp.0'). Run --list-layers to see options.")
    parser.add_argument("--scenario-id", default=None,
                        help="Override the auto-generated run UUID")
    parser.add_argument("--device", default=None,
                        help="Torch device (default: cuda:0 if available, else cpu)")
    parser.add_argument("--actor-obs-key", default="actor",
                        help="Key in the obs TensorDict used as actor input")
    parser.add_argument("--motion-file", default=None,
                        help="Path to motion .npz for tracking tasks")
    parser.add_argument("--list-layers", action="store_true",
                        help="Print all hookable layer names in the actor and exit")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Imports that require the mjlab environment to be installed
    # ------------------------------------------------------------------
    from tensordict import TensorDict

    import mjlab.tasks  # noqa: F401 — registers all tasks
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        sys.exit(f"Checkpoint not found: {checkpoint_path}")

    # ------------------------------------------------------------------
    # Build env (num_envs=1)
    # ------------------------------------------------------------------
    print(f"[1/4] Loading env config for task: {args.task}")
    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = 1

    # Tracking tasks need a motion file.
    if args.motion_file is not None:
        from mjlab.tasks.tracking.mdp import MotionCommandCfg
        motion_cmd = env_cfg.commands.get("motion")
        if isinstance(motion_cmd, MotionCommandCfg):
            motion_cmd.motion_file = args.motion_file
            print(f"  Motion file: {args.motion_file}")

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    agent_cfg = load_rl_cfg(args.task)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ------------------------------------------------------------------
    # Load runner + policy
    # ------------------------------------------------------------------
    print(f"[2/4] Loading checkpoint: {checkpoint_path.name}")
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(vec_env, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint_path),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )

    actor_model: torch.nn.Module = runner.alg.actor  # the MLPModel we hook
    rsl_policy = runner.get_inference_policy(device=device)  # callable: TensorDict → action

    # ------------------------------------------------------------------
    # Optional: list layer names and exit
    # ------------------------------------------------------------------
    if args.list_layers:
        print("\nHookable layers in the actor MLPModel:")
        for name, mod in actor_model.named_modules():
            if name:  # skip root ""
                print(f"  {name!r:40s}  {mod.__class__.__name__}")
        vec_env.close()
        return

    # ------------------------------------------------------------------
    # policy_fn: numpy obs → numpy action (hooks fire inside rsl_policy)
    # ------------------------------------------------------------------
    actor_obs_key = args.actor_obs_key

    def policy_fn(obs_np):
        """Convert numpy obs to TensorDict, call rsl_policy, return numpy action."""
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        obs_td = TensorDict({actor_obs_key: obs_t}, batch_size=[1])
        with torch.no_grad():
            action_t = rsl_policy(obs_td)
        return action_t.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # context_fn: record named obs each step (generic fallback)
    # ------------------------------------------------------------------
    context_fn = _build_context_fn(obs_names=None)

    # ------------------------------------------------------------------
    # Collect
    # ------------------------------------------------------------------
    print(f"[3/4] Collecting {args.steps} steps (layer: {args.layer!r})")

    from interlatent.api import LatentDB
    from interlatent.collectors.trajectory_collector import TrajectoryCollector

    db = LatentDB(f"sqlite:///{args.db}")
    shim_env = _MjlabGymShim(vec_env, actor_obs_key=actor_obs_key)

    collector = TrajectoryCollector(
        db,
        hook_layers=[args.layer],
    )

    run_info = collector.run(
        model=actor_model,
        env=shim_env,
        steps=args.steps,
        scenario_id=args.scenario_id,
        policy_fn=policy_fn,
        context_fn=context_fn,
        tags={"task": args.task, "checkpoint": checkpoint_path.name},
    )

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n[4/4] Collection complete.")
    print(f"  Run ID:   {run_info.run_id}")
    print(f"  Short ID: {run_info.run_id[:8]}")
    print(f"  Database: {args.db}")
    print(f"  Layer:    {args.layer}")
    print("\nNext: run the Interlatent pipeline:")
    print("  cd ../Interlatent-Robotics/pipeline")
    print("  python mvp.py process \\")
    print(f"      --db ../../mjlab_fork/{args.db} \\")
    print(f"      --run-id {run_info.run_id[:8]} \\")
    print(f"      --layer {args.layer} \\")
    print("      --k 32")

    vec_env.close()


if __name__ == "__main__":
    main()
