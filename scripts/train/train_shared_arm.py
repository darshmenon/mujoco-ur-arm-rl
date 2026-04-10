import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from envs.shared_arm_env import SharedArmBatchVecEnv, SharedArmPickPlaceEnv


LOG_ROOT = "logs/shared_arm"
MODEL_ROOT = "models/shared_arm"


def parse_args():
    parser = argparse.ArgumentParser(description="Train one reusable per-arm UR5e SAC policy.")
    parser.add_argument("--arms", type=int, default=8, help="Scene arm count used while sampling local arm tasks.")
    parser.add_argument("--n-envs", type=int, default=4, help="Parallel single-arm envs, or scene count with --all-arms-samples.")
    parser.add_argument(
        "--arm-index",
        type=int,
        default=None,
        help="Pin training to one arm index. Default samples a random arm each episode.",
    )
    parser.add_argument(
        "--all-arms-samples",
        action="store_true",
        help="Use every arm in each scene as a shared-policy sample every MuJoCo step.",
    )
    parser.add_argument("--timesteps", type=int, default=500_000, help="Total training timesteps.")
    parser.add_argument("--eval-freq", type=int, default=5_000, help="Evaluation frequency in timesteps.")
    parser.add_argument("--eval-episodes", type=int, default=8, help="Evaluation episodes per eval run.")
    parser.add_argument("--checkpoint-freq", type=int, default=50_000, help="Checkpoint frequency in timesteps.")
    parser.add_argument("--status-freq", type=int, default=500, help="Heartbeat frequency for latest_status.json.")
    parser.add_argument("--device", type=str, default="auto", help="Torch device: auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--curriculum", choices=["none", "easy_grasp"], default="easy_grasp")
    parser.add_argument("--resume-model", type=str, default=None, help="Optional shared-arm SAC model zip to resume.")
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def make_env(arm_count, curriculum_mode, arm_index):
    def _init():
        return SharedArmPickPlaceEnv(
            arm_count=arm_count,
            curriculum_mode=curriculum_mode,
            arm_index=arm_index,
        )

    return _init


def callback_freq(target_timesteps, n_envs):
    return max(target_timesteps // max(n_envs, 1), 1)


def safe_metric(values, key):
    value = values.get(key)
    if value is None:
        return None
    return float(value)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


class StatusCallback(BaseCallback):
    def __init__(self, run_dir, status_freq, eval_callback=None):
        super().__init__()
        self._status_path = os.path.join(run_dir, "latest_status.json")
        self._status_freq = status_freq
        self._last_dump_step = -1
        self._eval_callback = eval_callback
        self._latest_info = None

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
            "ep_rew_mean": safe_metric(values, "rollout/ep_rew_mean"),
            "ep_len_mean": safe_metric(values, "rollout/ep_len_mean"),
            "actor_loss": safe_metric(values, "train/actor_loss"),
            "critic_loss": safe_metric(values, "train/critic_loss"),
            "ent_coef": safe_metric(values, "train/ent_coef"),
            "ent_coef_loss": safe_metric(values, "train/ent_coef_loss"),
            "eval_mean_reward": eval_mean_reward,
            "eval_mean_ep_length": safe_metric(values, "eval/mean_ep_length"),
            "env0_info": self._latest_info,
        }
        with open(self._status_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        print(
            f"[status] steps={payload['timesteps']}"
            f" ep_rew_mean={payload['ep_rew_mean']}"
            f" eval_mean_reward={payload['eval_mean_reward']}",
            flush=True,
        )

    def _on_training_start(self):
        self._dump_status()
        self._last_dump_step = int(self.model.num_timesteps)

    def _on_step(self):
        infos = self.locals.get("infos")
        if infos:
            self._latest_info = json_safe(dict(infos[0]))
        if self.model.num_timesteps - self._last_dump_step >= self._status_freq:
            self._dump_status()
            self._last_dump_step = int(self.model.num_timesteps)
        return True

    def _on_training_end(self):
        self._dump_status()


def main():
    args = parse_args()
    if args.resume_model and not os.path.exists(args.resume_model):
        raise FileNotFoundError(f"Resume model not found: {args.resume_model}")

    os.makedirs(LOG_ROOT, exist_ok=True)
    os.makedirs(MODEL_ROOT, exist_ok=True)

    run_name = args.run_name or f"shared_arm_{args.arms}arm_scene_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(LOG_ROOT, run_name)
    model_dir = os.path.join(MODEL_ROOT, run_name)
    tb_dir = os.path.join(run_dir, "tb")
    checkpoints_dir = os.path.join(model_dir, "checkpoints")
    for path in (run_dir, model_dir, tb_dir, checkpoints_dir):
        os.makedirs(path, exist_ok=True)

    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)
    with open(os.path.join(LOG_ROOT, "latest_run.txt"), "w", encoding="utf-8") as handle:
        handle.write(f"{run_dir}\n")

    effective_envs = args.n_envs * args.arms if args.all_arms_samples else args.n_envs

    if args.all_arms_samples:
        vec_env = SharedArmBatchVecEnv(
            arm_count=args.arms,
            scene_count=args.n_envs,
            curriculum_mode=args.curriculum,
        )
    else:
        vec_env = DummyVecEnv([make_env(args.arms, args.curriculum, args.arm_index) for _ in range(args.n_envs)])
    vec_env = VecMonitor(vec_env, filename=os.path.join(run_dir, "train_monitor.csv"))
    eval_env = DummyVecEnv([make_env(args.arms, args.curriculum, args.arm_index)])
    eval_env = VecMonitor(eval_env, filename=os.path.join(run_dir, "eval_monitor.csv"))

    n_actions = vec_env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions, dtype=np.float32),
        sigma=0.1 * np.ones(n_actions, dtype=np.float32),
    )

    if args.resume_model:
        print(f"Resuming shared-arm training from {args.resume_model}", flush=True)
        model = SAC.load(
            args.resume_model,
            env=vec_env,
            tensorboard_log=tb_dir,
            action_noise=action_noise,
            device=args.device,
        )
    else:
        model = SAC(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=tb_dir,
            learning_rate=3e-4,
            buffer_size=300_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            ent_coef="auto",
            target_entropy=-3,
            learning_starts=2_000,
            train_freq=2,
            gradient_steps=8,
            action_noise=action_noise,
            policy_kwargs=dict(net_arch=[256, 256, 256]),
            device=args.device,
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=run_dir,
        eval_freq=callback_freq(args.eval_freq, effective_envs),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )

    callbacks = CallbackList(
        [
            eval_callback,
            CheckpointCallback(
                save_freq=callback_freq(args.checkpoint_freq, effective_envs),
                save_path=checkpoints_dir,
                name_prefix="shared_arm",
            ),
            StatusCallback(run_dir, args.status_freq, eval_callback=eval_callback),
        ]
    )

    print(
        f"Training shared 1-arm SAC policy in a {args.arms}-arm scene with "
        f"{args.n_envs} scene/env units ({effective_envs} SAC envs). "
        f"All-arms-samples={'on' if args.all_arms_samples else 'off'}. "
        f"Run dir: {run_dir}",
        flush=True,
    )
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=False,
        log_interval=10,
        reset_num_timesteps=not bool(args.resume_model),
    )

    final_model_path = os.path.join(model_dir, "shared_arm_final")
    model.save(final_model_path)
    print(f"Training done. Final shared-arm model saved to {final_model_path}", flush=True)


if __name__ == "__main__":
    main()
