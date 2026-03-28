import sys
sys.path.insert(0, "/home/asimov/mujoco-ur-arm-rl")

import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import os

LOG_DIR = "logs/2arm"
MODEL_DIR = "models/2arm"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def build_2arm_spec():
    arm_l = mujoco.MjSpec.from_file("/home/asimov/mujoco_menagerie/universal_robots_ur5e/ur5e.xml")
    grip_l = mujoco.MjSpec.from_file("/home/asimov/mujoco_menagerie/robotiq_2f85/2f85.xml")
    arm_l.sites[0].attach_body(grip_l.worldbody.first_body(), "l_gr-", "")

    arm_r = mujoco.MjSpec.from_file("/home/asimov/mujoco_menagerie/universal_robots_ur5e/ur5e.xml")
    grip_r = mujoco.MjSpec.from_file("/home/asimov/mujoco_menagerie/robotiq_2f85/2f85.xml")
    arm_r.sites[0].attach_body(grip_r.worldbody.first_body(), "r_gr-", "")

    spec = mujoco.MjSpec()
    spec.option.gravity = [0, 0, -9.81]

    floor = spec.worldbody.add_geom()
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0, 0, 0.05]
    floor.rgba = [0.3, 0.3, 0.3, 1]

    light = spec.worldbody.add_light()
    light.pos = [0, 0, 2.5]

    for arm_spec, name, pos, ez in [
        (arm_l, "left",  [-0.75, 0.0, 0.0],   0.0),
        (arm_r, "right", [ 0.75, 0.0, 0.0], 180.0),
    ]:
        mount = spec.worldbody.add_body()
        mount.name = f"{name}_mount"
        mount.pos = pos
        mount.alt.euler = [0, 0, ez]
        base = mount.add_geom()
        base.type = mujoco.mjtGeom.mjGEOM_BOX
        base.size = [0.1, 0.1, 0.15]
        base.pos = [0, 0, -0.15]
        base.rgba = [0.2, 0.2, 0.2, 1]
        frame = mount.add_frame()
        frame.attach_body(arm_spec.worldbody.first_body(), f"{name}_", "")

    # Table
    table = spec.worldbody.add_body()
    table.pos = [0, 0, 0]
    tg = table.add_geom()
    tg.type = mujoco.mjtGeom.mjGEOM_BOX
    tg.size = [0.5, 0.35, 0.02]
    tg.rgba = [0.8, 0.6, 0.4, 1]

    # Items (2 per side)
    for name, pos, color in [
        ("obj_l0", [-0.3, -0.15, 0.045], [0.9, 0.1, 0.1, 1]),
        ("obj_l1", [-0.3,  0.15, 0.045], [0.1, 0.1, 0.9, 1]),
        ("obj_r0", [ 0.3, -0.15, 0.045], [0.1, 0.8, 0.1, 1]),
        ("obj_r1", [ 0.3,  0.15, 0.045], [0.9, 0.8, 0.1, 1]),
    ]:
        b = spec.worldbody.add_body()
        b.name = name
        b.pos = pos
        fj = b.add_freejoint()
        fj.name = f"{name}_j"
        g = b.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [0.025, 0.025, 0.025]
        g.rgba = color
        g.mass = 0.1
        g.friction = [2.0, 0.01, 0.001]

    # Central handover object
    obj = spec.worldbody.add_body()
    obj.name = "object"
    obj.pos = [-0.25, 0.0, 0.045]
    fj = obj.add_freejoint()
    fj.name = "object_j"
    og = obj.add_geom()
    og.type = mujoco.mjtGeom.mjGEOM_BOX
    og.size = [0.03, 0.03, 0.03]
    og.rgba = [1.0, 0.5, 0.0, 1]
    og.mass = 0.1
    og.friction = [2.0, 0.01, 0.001]

    # Drop zone
    dz = spec.worldbody.add_body()
    dz.name = "drop_zone"
    dz.pos = [0.3, 0.0, 0.022]
    dzg = dz.add_geom()
    dzg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    dzg.size = [0.06, 0.002, 0]
    dzg.rgba = [0.1, 0.9, 0.1, 0.4]
    dzg.contype = 0
    dzg.conaffinity = 0

    return spec


class TwoArmEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        spec = build_2arm_spec()
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        self._n_arm = 6
        self._n_grip = 1
        self._n_per = self._n_arm + self._n_grip
        self._n_ctrl = self._n_per * 2

        self._l_ee = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_attachment_site")
        self._r_ee = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_attachment_site")
        self._obj = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self._drop_pos = np.array([0.3, 0.0, 0.022])

        obs_dim = 6+6 + 6+6 + 3+3+3+3 + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self._n_ctrl,), dtype=np.float32)
        self._viewer = None

    def _obs(self):
        lq = self.data.qpos[:6].astype(np.float32)
        lv = self.data.qvel[:6].astype(np.float32)
        rq = self.data.qpos[self._n_per:self._n_per+6].astype(np.float32)
        rv = self.data.qvel[self._n_per:self._n_per+6].astype(np.float32)
        lee = self.data.site_xpos[self._l_ee].astype(np.float32)
        ree = self.data.site_xpos[self._r_ee].astype(np.float32)
        obj = self.data.xpos[self._obj].astype(np.float32)
        drop = self._drop_pos.astype(np.float32)
        grips = np.array([self.data.qpos[6], self.data.qpos[self._n_per+6]], dtype=np.float32)
        return np.concatenate([lq, lv, rq, rv, lee, ree, obj, drop, grips])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        obj_start = self._n_per * 2
        self.data.qpos[obj_start:obj_start+3] = [-0.25, 0.0, 0.045]
        self.data.qpos[obj_start+3:obj_start+7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), {}

    def step(self, action):
        self.data.ctrl[:6] = action[:6] * 0.5
        if self.model.nu > 6:
            self.data.ctrl[6] = (action[6] + 1) / 2 * 0.8
        self.data.ctrl[self._n_per:self._n_per+6] = action[self._n_per:self._n_per+6] * 0.5
        if self.model.nu > self._n_per+6:
            self.data.ctrl[self._n_per+6] = (action[self._n_per+6] + 1) / 2 * 0.8

        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        obj = self.data.xpos[self._obj]
        lee = self.data.site_xpos[self._l_ee]
        ree = self.data.site_xpos[self._r_ee]
        d_l = np.linalg.norm(lee - obj)
        d_r = np.linalg.norm(ree - obj)
        d_drop = np.linalg.norm(obj[:2] - self._drop_pos[:2])

        reward = -0.5 * min(d_l, d_r) - 1.0 * d_drop
        if d_drop < 0.06 and obj[2] < 0.06:
            reward += 10.0

        terminated = bool(d_drop < 0.05 and obj[2] < 0.06)
        truncated = bool(self.data.time > 15.0)

        if self.render_mode == "human" and self._viewer:
            self._viewer.sync()

        return self._obs(), reward, terminated, truncated, {"d_drop": d_drop}

    def render(self):
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def close(self):
        if self._viewer:
            self._viewer.close()
            self._viewer = None


class LiveViewerCallback(BaseCallback):
    def __init__(self, env, sync_every=5):
        super().__init__()
        self._env = env
        self._sync_every = sync_every
        self._viewer = None

    def _on_training_start(self):
        self._viewer = mujoco.viewer.launch_passive(self._env.model, self._env.data)

    def _on_step(self):
        if self.n_calls % self._sync_every == 0 and self._viewer:
            self._viewer.sync()
        return True

    def _on_training_end(self):
        if self._viewer:
            self._viewer.close()


env = TwoArmEnv()
eval_env = TwoArmEnv()

model = SAC(
    "MlpPolicy", env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=3e-4,
    buffer_size=200_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    ent_coef="auto",
)

live_cb = LiveViewerCallback(env, sync_every=5)
eval_cb = EvalCallback(eval_env, best_model_save_path=MODEL_DIR,
                       log_path=LOG_DIR, eval_freq=5_000,
                       n_eval_episodes=5, deterministic=True)

print("Training 2-arm handover with live viewer...")
model.learn(total_timesteps=500_000, callback=[live_cb, eval_cb], progress_bar=True)
model.save(f"{MODEL_DIR}/2arm_final")
print("Done.")
