# Project Concepts

## What This Project Is

This project trains robotic arms to perform pick-and-place tasks using reinforcement learning inside a physics simulator. The goal is to teach robots useful skills — reaching, grasping, lifting, and placing objects — entirely in simulation, then deploy the learned policies to real hardware via ROS2.

## The Simulator: MuJoCo

MuJoCo (Multi-Joint dynamics with Contact) is a physics engine built for robotics research. It simulates rigid body dynamics, contacts, and actuator physics accurately and fast enough to run thousands of training episodes per hour. Running this on real hardware would be impossible — a single drop could damage the robot, and each episode would take minutes instead of milliseconds.

Every 5 MuJoCo physics steps = 1 action step. This smooths out the control signal and keeps the simulation stable.

## The Robot: UR5e + Robotiq 2F-85

The Universal Robots UR5e is a 6-DOF collaborative robot arm widely used in research and industry. We attach a Robotiq 2F-85 two-finger parallel gripper to the end-effector, giving the arm the ability to grasp objects.

The UR5e has 6 joints (shoulder pan, shoulder lift, elbow, wrist 1, wrist 2, wrist 3). The gripper has 1 actuator but 8 internal joints in the MuJoCo model — this is important because it means each arm occupies 14 qpos slots, not 7. Getting this wrong causes the wrong joints to be set on reset, which was a bug we fixed.

All robot models come from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).

## The Algorithm: SAC

We use Soft Actor-Critic (SAC), an off-policy deep RL algorithm designed for continuous control.

- **Off-policy**: learns from a replay buffer of past transitions, not just the latest episode. This makes it much more sample-efficient than on-policy algorithms like PPO.
- **Entropy regularization**: SAC adds an entropy bonus to the reward, encouraging the agent to stay random (explore) early on and only commit to behaviors that are clearly better. The `ent_coef` (entropy coefficient) controls this — if it collapses to near zero too early, the agent stops exploring and gets stuck.
- **Continuous actions**: outputs 28 continuous numbers (6 joints + 1 gripper per arm × 4 arms), scaled to joint velocity commands.

Key hyperparameters we tuned:
- `target_entropy = -14` (half of action dim) — keeps exploration alive longer
- `learning_starts = 5000` — fills the buffer with random experience before training begins
- `train_freq = 4, gradient_steps = 4` — 4 gradient updates per 4 env steps

## The Phase-Based Reward System

The most important design decision in this project. Inspired by the `pickplace-rl-mobile` project.

The core problem with naive rewards (e.g. `-distance_to_drop`) is that the agent never sees success early on, so it has no signal to learn from. The fix is to break the task into phases with dense rewards at each stage:

### Phase 0 — Reach
Move the end-effector to the object's XY position at grasp height. Reward is based on **improvement per step** (delta distance × 100), not absolute distance. This means the agent gets a positive reward every time it moves closer, and a penalty (3× larger) when it moves away. Penalises closing the gripper during approach.

Transition condition: EE within 6cm XY and 4cm Z of object → +100 bonus.

### Phase 1 — Grasp
Close the gripper while the EE is at the object. Rewards gripper closing when within 5cm. Penalises opening the gripper when close. Touch-range bonuses at <10cm, <4cm, <3cm.

Transition condition: gripper >60% closed AND EE within 5cm → **+1000 bonus**.

### Phase 2 — Lift
Raise the EE to lift height (0.10m above table). Delta reward on Z distance. Penalises dropping the object (gripper opening).

Transition condition: within 5cm of lift height → +200 bonus.

### Phase 3 — Place
Move the object over the drop zone and open the gripper. Delta reward on XY distance to drop zone.

Success condition: within 8cm of drop zone AND gripper open → **+1000 bonus**, episode ends for this arm.

### Why Deltas Not Absolutes
Using `(prev_dist - curr_dist) × scale` instead of `-curr_dist × scale`:
- The agent starts getting reward immediately (any improvement = positive signal)
- The reward magnitude is consistent regardless of how far the arm starts
- Retreating is penalised 2-4× harder than advancing, discouraging oscillation

### Episode Length
`max_episode_steps = 600` (not time-based). Each episode is at most 600 action steps. This is much shorter than the previous 2001 step limit and forces the agent to learn efficient behavior.

## The 4-Arm Environment (URDualArmEnv)

Four UR5e arms arranged symmetrically around a central table:

```
left2 [-0.9,+0.5] → blue box  → drop at [-0.2,+0.35]
left1 [-0.9,-0.5] → red box   → drop at [-0.2,-0.35]
         [   central table + 6 obstacle boxes   ]
right1[+0.9,-0.5] → green box → drop at [+0.2,-0.35]
right2[+0.9,+0.5] → yellow box→ drop at [+0.2,+0.35]
```

Each arm is fully independent — it has its own object, its own drop zone, its own phase counter, and its own reward. The episode ends when all 4 arms have successfully placed their objects.

Arms start in a "ready pose" (elbow bent, EE pointing down toward their object at ~0.25m distance) instead of the default upright pose which puts the EE 1.3m away.

## Observation Space

Per arm (23 values × 4 arms = 92 total):
- Joint positions (6) + velocities (6)
- End-effector position in world frame (3)
- Object position (3)
- Drop zone position (3)
- Gripper finger joint position (1)
- Current phase (1)

## Action Space

28 values (7 per arm × 4 arms):
- 6 joint velocity commands in [-1, 1], scaled by 0.5 rad/s
- 1 gripper command: -1=open, +1=close, mapped to [0, 0.8]

## ROS2 Integration

A ROS2 inference node (`ros2/ur_policy_node.py`) loads a trained SAC model and publishes joint trajectory commands to a real UR5e. It subscribes to `/joint_states` and publishes to the UR's trajectory controller at 10Hz.

## Future Work

- Isaac Lab: GPU-accelerated parallel training (1000× faster than CPU MuJoCo)
- Sim-to-real: domain randomization on object mass, friction, and visual appearance
- Multi-task: single policy that handles reach, pick, place, and handover
- Curriculum: start with objects close to the drop zone, gradually increase difficulty
