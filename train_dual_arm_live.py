import sys
sys.path.insert(0, "/home/asimov/mujoco-ur-arm-rl")

from envs.ur_dual_arm_env import URDualArmEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.noise import NormalActionNoise
import mujoco.viewer
import numpy as np
import os

LOG_DIR = "logs/dual_arm_live"
MODEL_DIR = "models/dual_arm"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

N_ENVS = 4  # parallel envs — 4× more experience per second


class LiveViewerCallback(BaseCallback):
    def __init__(self, render_every=20):
        super().__init__()
        self._render_every = render_every
        self._viewer = None

    def _on_training_start(self):
        # Connect directly to the first training env so we see actual arm movement
        env0 = self.training_env.envs[0]
        self._viewer = mujoco.viewer.launch_passive(env0.model, env0.data)

    def _on_step(self):
        if self.n_calls % self._render_every == 0 and self._viewer:
            self._viewer.sync()
        return True

    def _on_training_end(self):
        if self._viewer:
            self._viewer.close()


def make_env():
    def _init():
        return URDualArmEnv()
    return _init


# Parallel envs for faster sampling
vec_env  = DummyVecEnv([make_env() for _ in range(N_ENVS)])
vec_env  = VecMonitor(vec_env)
eval_env = URDualArmEnv()

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=MODEL_DIR,
    log_path=LOG_DIR,
    eval_freq=5_000,
    n_eval_episodes=5,
    deterministic=True,
)
live_cb = LiveViewerCallback(render_every=20)

# Action noise for extra exploration on top of SAC entropy
n_actions = vec_env.action_space.shape[0]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions),
    sigma=0.2 * np.ones(n_actions)
)

model = SAC(
    "MlpPolicy",
    vec_env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=3e-4,
    buffer_size=500_000,
    batch_size=512,
    gamma=0.99,
    tau=0.005,
    ent_coef="auto",
    target_entropy=-10,       # more exploration (less negative = higher entropy target)
    learning_starts=2_000,
    train_freq=2,
    gradient_steps=8,         # more gradient steps per rollout = faster learning
    action_noise=action_noise,
    policy_kwargs=dict(
        net_arch=[256, 256, 256],  # deeper network for complex 4-arm task
    ),
)

print(f"Training with {N_ENVS} parallel envs, target_entropy=-10, action noise σ=0.2")
model.learn(
    total_timesteps=1_000_000,
    callback=[live_cb, eval_cb],
    progress_bar=True,
)
model.save(f"{MODEL_DIR}/ur5e_4arm_final")
print("Done.")
