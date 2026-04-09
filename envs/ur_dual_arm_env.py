import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces

TABLE_Z = 0.02
OBJECT_Z = 0.045
LIFT_Z = 0.10
Y_SPACING = 1.0

ARM_MODEL_PATH = "/home/asimov/mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
GRIPPER_MODEL_PATH = "/home/asimov/mujoco_menagerie/robotiq_2f85/2f85.xml"

SIDE_CFG = {
    "left": {
        "base_x": -0.9,
        "table_x": -1.18,
        "object_x": -1.05,
        "drop_x": -1.25,
        "euler_z": 0.0,
        "ready_pose": np.array([np.pi, -1.0, 1.5, -1.57, -1.57, 0.0], dtype=np.float64),
    },
    "right": {
        "base_x": 0.9,
        "table_x": 0.58,
        "object_x": 0.45,
        "drop_x": 0.68,
        "euler_z": 180.0,
        "ready_pose": np.array([0.0, -1.0, 1.5, -1.57, -1.57, 0.0], dtype=np.float64),
    },
}

OBJECT_COLORS = [
    [0.9, 0.1, 0.1, 1.0],
    [0.1, 0.1, 0.9, 1.0],
    [0.1, 0.8, 0.1, 1.0],
    [0.9, 0.8, 0.1, 1.0],
    [0.9, 0.2, 0.8, 1.0],
    [0.1, 0.8, 0.8, 1.0],
    [0.95, 0.45, 0.1, 1.0],
    [0.5, 0.9, 0.1, 1.0],
]

OBSTACLE_COLORS = [
    [0.6, 0.3, 0.3, 1.0],
    [0.3, 0.6, 0.3, 1.0],
    [0.3, 0.3, 0.6, 1.0],
    [0.6, 0.6, 0.2, 1.0],
    [0.6, 0.2, 0.6, 1.0],
    [0.2, 0.6, 0.6, 1.0],
    [0.8, 0.4, 0.2, 1.0],
    [0.4, 0.8, 0.2, 1.0],
]


def _centered_y_positions(count_per_side):
    offset = (count_per_side - 1) / 2.0
    return [float((idx - offset) * Y_SPACING) for idx in range(count_per_side)]


def _build_arm_cfg(arm_count):
    if arm_count < 2 or arm_count % 2 != 0:
        raise ValueError("arm_count must be an even number >= 2")

    cfg = []
    y_positions = _centered_y_positions(arm_count // 2)

    for side in ("left", "right"):
        side_cfg = SIDE_CFG[side]
        for arm_idx, y_pos in enumerate(y_positions, start=1):
            color_idx = len(cfg) % len(OBJECT_COLORS)
            name = f"{side}{arm_idx}"
            cfg.append(
                {
                    "name": name,
                    "side": side,
                    "base_pos": [side_cfg["base_x"], y_pos, 0.0],
                    "euler_z": side_cfg["euler_z"],
                    "table_pos": [side_cfg["table_x"], y_pos],
                    "table_size": [0.28, 0.20],
                    "obj_pos": [side_cfg["object_x"], y_pos, OBJECT_Z],
                    "obj_color": OBJECT_COLORS[color_idx],
                    "drop_pos": [side_cfg["drop_x"], y_pos, TABLE_Z],
                    "ready_pose": side_cfg["ready_pose"].copy(),
                    "obstacle_color": OBSTACLE_COLORS[color_idx % len(OBSTACLE_COLORS)],
                }
            )

    return cfg


class URDualArmEnv(gym.Env):
    """
    Symmetric multi-arm UR5e task.
    Each arm gets its own table, object, and drop zone.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, arm_count=4):
        self.render_mode = render_mode
        self.arm_cfg = _build_arm_cfg(arm_count)

        spec = mujoco.MjSpec()
        spec.option.gravity = [0, 0, -9.81]

        tex = spec.add_texture()
        tex.name = "groundplane"
        tex.type = mujoco.mjtTexture.mjTEXTURE_2D
        tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
        tex.rgb1 = [0.2, 0.3, 0.4]
        tex.rgb2 = [0.1, 0.2, 0.3]
        tex.width = 300
        tex.height = 300

        mat = spec.add_material()
        mat.name = "groundplane"
        mat.textures = ["groundplane"] + [""] * 9
        mat.texuniform = True

        floor = spec.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = [0, 0, 0.05]
        floor.material = "groundplane"

        spec.worldbody.add_light().pos = [0, 0, 2.5]

        def attach_arm(arm_name, pos, euler_z):
            arm_spec = mujoco.MjSpec.from_file(ARM_MODEL_PATH)
            grip_spec = mujoco.MjSpec.from_file(GRIPPER_MODEL_PATH)
            arm_spec.sites[0].attach_body(grip_spec.worldbody.first_body(), f"{arm_name}_gr-", "")

            mount = spec.worldbody.add_body()
            mount.name = f"{arm_name}_mount"
            mount.pos = pos
            mount.alt.euler = [0, 0, euler_z]

            base = mount.add_geom()
            base.name = f"{arm_name}_base"
            base.type = mujoco.mjtGeom.mjGEOM_BOX
            base.size = [0.1, 0.1, 0.15]
            base.pos = [0, 0, -0.15]
            base.rgba = [0.2, 0.2, 0.2, 1.0]

            frame = mount.add_frame()
            frame.attach_body(arm_spec.worldbody.first_body(), f"{arm_name}_", "")

        for cfg in self.arm_cfg:
            attach_arm(cfg["name"], cfg["base_pos"], cfg["euler_z"])

        for arm_idx, cfg in enumerate(self.arm_cfg):
            table = spec.worldbody.add_body()
            table.name = f"table_{cfg['name']}"
            table.pos = [cfg["table_pos"][0], cfg["table_pos"][1], 0.0]

            table_geom = table.add_geom()
            table_geom.name = f"table_{cfg['name']}_geom"
            table_geom.type = mujoco.mjtGeom.mjGEOM_BOX
            table_geom.size = [cfg["table_size"][0], cfg["table_size"][1], TABLE_Z]
            table_geom.rgba = [0.8, 0.6, 0.4, 1.0]

            extra = spec.worldbody.add_body()
            extra.name = f"extra_{cfg['name']}"
            extra.pos = [cfg["table_pos"][0], cfg["table_pos"][1], TABLE_Z + 0.025]

            extra_joint = extra.add_freejoint()
            extra_joint.name = f"extra_{cfg['name']}_joint"

            extra_geom = extra.add_geom()
            extra_geom.name = f"extra_{cfg['name']}_geom"
            extra_geom.type = mujoco.mjtGeom.mjGEOM_BOX
            extra_geom.size = [0.02, 0.02, 0.02]
            extra_geom.rgba = cfg["obstacle_color"]
            extra_geom.mass = 0.05
            extra_geom.friction = [2.0, 0.01, 0.001]

            obj = spec.worldbody.add_body()
            obj.name = f"obj_{cfg['name']}"
            obj.pos = cfg["obj_pos"]

            obj_joint = obj.add_freejoint()
            obj_joint.name = f"obj_{cfg['name']}_joint"

            obj_geom = obj.add_geom()
            obj_geom.name = f"obj_{cfg['name']}_geom"
            obj_geom.type = mujoco.mjtGeom.mjGEOM_BOX
            obj_geom.size = [0.025, 0.025, 0.025]
            obj_geom.rgba = cfg["obj_color"]
            obj_geom.mass = 0.1
            obj_geom.friction = [2.0, 0.01, 0.001]

            drop = spec.worldbody.add_body()
            drop.name = f"drop_{cfg['name']}"
            drop.pos = cfg["drop_pos"]

            drop_geom = drop.add_geom()
            drop_geom.name = f"drop_{cfg['name']}_geom"
            drop_geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            drop_geom.size = [0.055, 0.002, 0]
            drop_geom.rgba = [0.1, 0.9, 0.1, 0.5]
            drop_geom.contype = 0
            drop_geom.conaffinity = 0

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        self._n_arm = 6
        self._n_arms = len(self.arm_cfg)
        self._n_ctrl = self.model.nu
        self.arm_names = [cfg["name"] for cfg in self.arm_cfg]

        self._arm_qpos_adr = [
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_shoulder_pan_joint")
            ]
            for name in self.arm_names
        ]
        self._arm_qvel_adr = [
            self.model.jnt_dofadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_shoulder_pan_joint")
            ]
            for name in self.arm_names
        ]
        self._grip_qpos_adr = [
            self.model.jnt_qposadr[
                mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"{name}_{name}_gr-right_driver_joint",
                )
            ]
            for name in self.arm_names
        ]
        self._ee_sites = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{name}_attachment_site")
            for name in self.arm_names
        ]
        self._obj_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obj_{name}")
            for name in self.arm_names
        ]
        self._drop_positions = [np.array(cfg["drop_pos"], dtype=np.float64) for cfg in self.arm_cfg]
        self._obj_init_pos = [np.array(cfg["obj_pos"], dtype=np.float64) for cfg in self.arm_cfg]
        self._obj_qpos_adr = [
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"obj_{name}_joint")
            ]
            for name in self.arm_names
        ]

        obs_dim = self._n_arms * 23
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self._n_ctrl,), dtype=np.float32)

        self._viewer = None
        self._phase = [0] * self._n_arms
        self._prev_dist = [None] * self._n_arms
        self._grasped = [False] * self._n_arms
        self._episode_steps = 0
        self.max_episode_steps = 500 * self._n_arms

    def _get_obs(self):
        parts = []
        for i in range(self._n_arms):
            qpos_adr = self._arm_qpos_adr[i]
            qvel_adr = self._arm_qvel_adr[i]
            parts.append(self.data.qpos[qpos_adr:qpos_adr + self._n_arm].astype(np.float32))
            parts.append(self.data.qvel[qvel_adr:qvel_adr + self._n_arm].astype(np.float32))
            parts.append(self.data.site_xpos[self._ee_sites[i]].astype(np.float32))
            parts.append(self.data.xpos[self._obj_ids[i]].astype(np.float32))
            parts.append(self._drop_positions[i].astype(np.float32))
            parts.append(np.array([self.data.qpos[self._grip_qpos_adr[i]]], dtype=np.float32))
            parts.append(np.array([float(self._phase[i])], dtype=np.float32))
        return np.concatenate(parts)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        for i, cfg in enumerate(self.arm_cfg):
            qpos_adr = self._arm_qpos_adr[i]
            self.data.qpos[qpos_adr:qpos_adr + self._n_arm] = cfg["ready_pose"]

        for i, init_pos in enumerate(self._obj_init_pos):
            obj_qpos_adr = self._obj_qpos_adr[i]
            self.data.qpos[obj_qpos_adr:obj_qpos_adr + 3] = init_pos
            self.data.qpos[obj_qpos_adr + 3:obj_qpos_adr + 7] = [1, 0, 0, 0]

        self._phase = [0] * self._n_arms
        self._prev_dist = [None] * self._n_arms
        self._grasped = [False] * self._n_arms
        self._episode_steps = 0

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def _arm_reward(self, i):
        ee = self.data.site_xpos[self._ee_sites[i]].copy()
        obj = self.data.xpos[self._obj_ids[i]].copy()
        drop = self._drop_positions[i]
        init_obj = self._obj_init_pos[i]
        grip = float(self.data.qpos[self._grip_qpos_adr[i]])
        qvel_adr = self._arm_qvel_adr[i]

        ee_to_obj = float(np.linalg.norm(ee - obj))
        obj_to_drop_xy = float(np.linalg.norm(obj[:2] - drop[:2]))
        obj_lift = float(obj[2] - init_obj[2])
        joint_vel_penalty = 0.01 * float(np.sum(np.abs(self.data.qvel[qvel_adr:qvel_adr + self._n_arm])))

        reward = 0.0
        terminated_arm = False
        phase = self._phase[i]

        if phase == 0:
            grasp_z = float(obj[2])
            dist_xy = float(np.linalg.norm(ee[:2] - obj[:2]))
            dist_z = abs(ee[2] - grasp_z)
            dist = dist_xy + dist_z

            reward += 5.0 / (1.0 + dist * 10.0)

            if self._prev_dist[i] is not None:
                delta = self._prev_dist[i] - dist
                reward += delta * 300.0
            self._prev_dist[i] = dist

            if dist_xy < 0.20:
                reward += 15.0 * (1.0 - dist_xy / 0.20)
            if dist_z < 0.10:
                reward += 10.0 * (1.0 - dist_z / 0.10)
            if dist_xy < 0.08 and dist_z < 0.04:
                reward += 40.0

            if grip > 0.5:
                reward -= 2.0

            if dist_xy < 0.06 and dist_z < 0.04:
                self._phase[i] = 1
                self._prev_dist[i] = None
                reward += 500.0

        elif phase == 1:
            reward += 5.0 / (1.0 + ee_to_obj * 10.0)

            if self._prev_dist[i] is not None:
                delta = self._prev_dist[i] - ee_to_obj
                reward += delta * 300.0
            self._prev_dist[i] = ee_to_obj

            if ee_to_obj < 0.10:
                reward += 20.0 * (1.0 - ee_to_obj / 0.10)
            if ee_to_obj < 0.05:
                reward += 40.0 * (1.0 - ee_to_obj / 0.05)
            if ee_to_obj < 0.03:
                reward += 30.0

            reward += 20.0 * grip
            if ee_to_obj < 0.10:
                reward += 30.0 * grip
            if grip > 0.4 and ee_to_obj < 0.06:
                reward += 50.0 * grip
            if grip < 0.2 and ee_to_obj < 0.05:
                reward -= 20.0

            if grip > 0.35 and ee_to_obj < 0.08:
                reward += max(0.0, obj_lift) * 800.0
                if obj_lift > 0.01:
                    reward += 100.0

            if grip > 0.4 and ee_to_obj < 0.08 and obj_lift > 0.03:
                self._grasped[i] = True
                self._phase[i] = 2
                self._prev_dist[i] = None
                reward += 2000.0

        elif phase == 2:
            if grip < 0.3:
                reward -= 20.0
                self._grasped[i] = False

            dist_z = abs(obj[2] - LIFT_Z)
            if self._prev_dist[i] is not None:
                delta = self._prev_dist[i] - dist_z
                reward += delta * 150.0 if delta > 0 else delta * 250.0
            self._prev_dist[i] = dist_z

            reward += max(0.0, obj_lift) * 120.0
            reward -= ee_to_obj * 10.0

            if obj_lift < 0.01:
                reward -= 40.0
            if dist_z < 0.08:
                reward += 8.0 * (1.0 - dist_z / 0.08)

            if dist_z < 0.05 and obj_lift > 0.03:
                self._phase[i] = 3
                self._prev_dist[i] = None
                reward += 200.0

        elif phase == 3:
            if grip < 0.1 and obj_to_drop_xy > 0.10:
                reward -= 40.0

            if self._prev_dist[i] is not None:
                delta = self._prev_dist[i] - obj_to_drop_xy
                reward += delta * 80.0
            self._prev_dist[i] = obj_to_drop_xy

            reward += 5.0 / (1.0 + obj_to_drop_xy * 10.0)
            if obj_to_drop_xy < 0.10:
                reward += 25.0 * (1.0 - obj_to_drop_xy / 0.10)

            if obj_to_drop_xy < 0.08 and grip < 0.1 and obj[2] < init_obj[2] + 0.01:
                self._grasped[i] = False
                reward += 1000.0
                terminated_arm = True

        reward -= joint_vel_penalty
        return reward, terminated_arm, phase

    def step(self, action):
        self._episode_steps += 1

        for i in range(self._n_arms):
            start = i * 7
            self.data.ctrl[start:start + self._n_arm] = action[start:start + self._n_arm] * 0.5
            grip_idx = start + self._n_arm
            if grip_idx < self.model.nu:
                self.data.ctrl[grip_idx] = (action[grip_idx] + 1.0) / 2.0 * 0.8

        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        total_reward = 0.0
        info = {}
        all_done = True

        for i, name in enumerate(self.arm_names):
            reward, arm_done, phase = self._arm_reward(i)
            total_reward += reward
            if not arm_done:
                all_done = False

            obj = self.data.xpos[self._obj_ids[i]].copy()
            info[f"{name}_phase"] = phase
            info[f"{name}_done"] = arm_done
            info[f"{name}_obj_height"] = float(obj[2])
            info[f"{name}_obj_to_drop"] = float(np.linalg.norm(obj[:2] - self._drop_positions[i][:2]))

        terminated = all_done
        truncated = self._episode_steps >= self.max_episode_steps

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), total_reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
