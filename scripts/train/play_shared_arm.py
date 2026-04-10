import argparse
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from envs.shared_arm_env import LOCAL_ACTION_DIM, LOCAL_OBS_DIM
from envs.ur_dual_arm_env import URDualArmEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Apply one shared arm policy to every arm in a scene.")
    parser.add_argument("--model", required=True, help="Path to shared-arm SAC model zip.")
    parser.add_argument("--arms", type=int, default=8, help="Even number of arms in the scene.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--curriculum", choices=["none", "easy_grasp"], default="easy_grasp")
    parser.add_argument("--viewer", action="store_true", help="Show MuJoCo viewer.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap per episode.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional delay per env step.")
    return parser.parse_args()


def local_obs(full_obs, arm_index):
    start = arm_index * LOCAL_OBS_DIM
    return full_obs[start:start + LOCAL_OBS_DIM].astype(np.float32)


def main():
    args = parse_args()
    env = URDualArmEnv(
        render_mode="human" if args.viewer else None,
        arm_count=args.arms,
        curriculum_mode=args.curriculum,
    )
    model = SAC.load(args.model, device=args.device)

    try:
        for episode in range(args.episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            terminated = False
            truncated = False
            steps = 0

            while not terminated and not truncated:
                if args.max_steps is not None and steps >= args.max_steps:
                    break

                full_action = np.zeros(env.action_space.shape, dtype=np.float32)
                for arm_index in range(len(env.arm_names)):
                    action, _ = model.predict(
                        local_obs(obs, arm_index),
                        deterministic=args.deterministic,
                    )
                    start = arm_index * LOCAL_ACTION_DIM
                    full_action[start:start + LOCAL_ACTION_DIM] = action

                obs, reward, terminated, truncated, info = env.step(full_action)
                total_reward += float(reward)
                steps += 1

                if args.viewer:
                    env.render()
                if args.sleep > 0:
                    time.sleep(args.sleep)

            print(
                f"episode={episode + 1} steps={steps} reward={total_reward:.3f} "
                f"terminated={terminated} truncated={truncated}",
                flush=True,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
