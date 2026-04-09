import argparse
import json
import os
import sys
from datetime import datetime

import mujoco.viewer
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

sys.path.insert(0, "/home/asimov/mujoco-ur-arm-rl")

from envs.ur_dual_arm_env import URDualArmEnv


LOG_ROOT = "logs/multi_arm"
MODEL_ROOT = "models/multi_arm"


def parse_args():
    parser = argparse.ArgumentParser(description="Train symmetric multi-arm UR5e SAC policies.")
    parser.add_argument("--arms", type=int, default=4, help="Even number of arms to train, e.g. 4 or 8.")
    parser.add_argument("--n-envs", type=int, default=4, help="Parallel training environments.")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total training timesteps.")
    parser.add_argument("--eval-freq", type=int, default=5_000, help="Evaluation frequency in timesteps.")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Evaluation episodes per eval run.")
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50_000,
        help="Checkpoint frequency in timesteps.",
    )
    parser.add_argument(
        "--status-freq",
        type=int,
        default=500,
        help="Heartbeat frequency in timesteps for latest_status.json.",
    )
    parser.add_argument("--viewer", action="store_true", help="Launch the MuJoCo viewer.")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run directory name. Defaults to a timestamped name.",
    )
    return parser.parse_args()


def make_env(arm_count):
    def _init():
        return URDualArmEnv(arm_count=arm_count)

    return _init


def unwrap_first_env(vec_env):
    current = vec_env
    while hasattr(current, "venv"):
        current = current.venv
    return current.envs[0]


def safe_metric(values, key):
    value = values.get(key)
    if value is None:
        return None
    return float(value)


class LiveViewerCallback(BaseCallback):
    def __init__(self, render_every=20):
        super().__init__()
        self._render_every = render_every
        self._viewer = None

    def _on_training_start(self):
        env0 = unwrap_first_env(self.training_env)
        self._viewer = mujoco.viewer.launch_passive(env0.model, env0.data)

    def _on_step(self):
        if self._viewer is not None and self.n_calls % self._render_every == 0:
            self._viewer.sync()
        return True

    def _on_training_end(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


class StatusCallback(BaseCallback):
    def __init__(self, run_dir, status_freq, eval_callback=None):
        super().__init__()
        self._run_dir = run_dir
        self._status_path = os.path.join(run_dir, "latest_status.json")
        self._status_freq = status_freq
        self._last_dump_step = -1
        self._eval_callback = eval_callback

    def _dump_status(self):
        values = getattr(self.logger, "name_to_value", {})
        eval_mean_reward = safe_metric(values, "eval/mean_reward")
        if eval_mean_reward is None and self._eval_callback is not None:
            if np.isfinite(self._eval_callback.last_mean_reward):
                eval_mean_reward = float(self._eval_callback.last_mean_reward)

        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "timesteps": int(self.model.num_timesteps),
            "fps": safe_metric(values, "time/fps"),
            "episodes": safe_metric(values, "time/episodes"),
            "ep_rew_mean": safe_metric(values, "rollout/ep_rew_mean"),
            "ep_len_mean": safe_metric(values, "rollout/ep_len_mean"),
            "actor_loss": safe_metric(values, "train/actor_loss"),
            "critic_loss": safe_metric(values, "train/critic_loss"),
            "ent_coef": safe_metric(values, "train/ent_coef"),
            "ent_coef_loss": safe_metric(values, "train/ent_coef_loss"),
            "eval_mean_reward": eval_mean_reward,
            "eval_mean_ep_length": safe_metric(values, "eval/mean_ep_length"),
        }

        with open(self._status_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

        status_line = (
            f"[status] steps={payload['timesteps']}"
            f" fps={payload['fps']}"
            f" ep_rew_mean={payload['ep_rew_mean']}"
            f" eval_mean_reward={payload['eval_mean_reward']}"
        )
        print(status_line, flush=True)

    def _on_training_start(self):
        self._dump_status()
        self._last_dump_step = int(self.model.num_timesteps)

    def _on_step(self):
        if self.model.num_timesteps - self._last_dump_step >= self._status_freq:
            self._dump_status()
            self._last_dump_step = int(self.model.num_timesteps)
        return True

    def _on_training_end(self):
        self._dump_status()


def build_run_name(args):
    if args.run_name:
        return args.run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "viewer" if args.viewer else "headless"
    return f"{args.arms}arm_{mode}_{timestamp}"


def callback_freq(target_timesteps, n_envs):
    return max(target_timesteps // max(n_envs, 1), 1)


def main():
    args = parse_args()

    os.makedirs(LOG_ROOT, exist_ok=True)
    os.makedirs(MODEL_ROOT, exist_ok=True)

    run_name = build_run_name(args)
    run_dir = os.path.join(LOG_ROOT, run_name)
    model_dir = os.path.join(MODEL_ROOT, run_name)
    tb_dir = os.path.join(run_dir, "tb")
    checkpoints_dir = os.path.join(model_dir, "checkpoints")

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)

    with open(os.path.join(LOG_ROOT, "latest_run.txt"), "w", encoding="utf-8") as handle:
        handle.write(f"{run_dir}\n")

    vec_env = DummyVecEnv([make_env(args.arms) for _ in range(args.n_envs)])
    vec_env = VecMonitor(vec_env, filename=os.path.join(run_dir, "train_monitor.csv"))
    eval_env = DummyVecEnv([make_env(args.arms)])
    eval_env = VecMonitor(eval_env, filename=os.path.join(run_dir, "eval_monitor.csv"))

    n_actions = vec_env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions, dtype=np.float32),
        sigma=0.1 * np.ones(n_actions, dtype=np.float32),
    )

    model = SAC(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=tb_dir,
        learning_rate=3e-4,
        buffer_size=500_000,
        batch_size=512,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        target_entropy=-6,
        learning_starts=2_000,
        train_freq=2,
        gradient_steps=16,
        action_noise=action_noise,
        policy_kwargs=dict(net_arch=[256, 256, 256]),
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=run_dir,
        eval_freq=callback_freq(args.eval_freq, args.n_envs),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )

    callbacks = [
        eval_callback,
        CheckpointCallback(
            save_freq=callback_freq(args.checkpoint_freq, args.n_envs),
            save_path=checkpoints_dir,
            name_prefix=f"ur5e_{args.arms}arm",
        ),
        StatusCallback(run_dir=run_dir, status_freq=args.status_freq, eval_callback=eval_callback),
    ]

    if args.viewer:
        callbacks.append(LiveViewerCallback())

    print(
        f"Training {args.arms}-arm SAC with {args.n_envs} envs. "
        f"Viewer={'on' if args.viewer else 'off'}. Run dir: {run_dir}",
        flush=True,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=CallbackList(callbacks),
        progress_bar=False,
        log_interval=10,
    )

    final_model_path = os.path.join(model_dir, f"ur5e_{args.arms}arm_final")
    model.save(final_model_path)
    print(f"Training done. Final model saved to {final_model_path}", flush=True)


if __name__ == "__main__":
    main()
