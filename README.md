# mujoco-ur-arm-rl

Reinforcement learning training environment for the Universal Robots UR5e arm using MuJoCo, with a Robotiq 2F-85 gripper and graspable objects.

![Simulation Preview](assets/sim_preview.png)

## Overview

- **Robot**: UR5e + Robotiq 2F-85 gripper
- **Task**: Reach / pick-and-place / cooperative handover
- **Algorithm**: SAC (Stable-Baselines3)
- **Simulator**: MuJoCo 3.x

## Environments

| Env | Task | Arms |
|-----|------|------|
| `URReachEnv` | Move end-effector to target | 1 |
| `URPickPlaceEnv` | Pick box and place at target | 1 |
| `URDualArmEnv` | Symmetric multi-arm pick / lift / place | 4+ even |
| `SharedArmPickPlaceEnv` | Reusable local policy extracted from symmetric arms | 1 (local view) |

## Multi-Arm Training

![4-Arm Training](assets/4arm_training.png)

Four or more UR5e arms with Robotiq 2F-85 grippers arranged symmetrically. Each arm gets its own table, object, and drop zone, and the layout scales to 8 arms in the same pattern.

The reward is contact-gated and phase-based (reach → grasp → lift → place). The policy is rewarded for approaching the object, but grasp/lift progress only counts once the object contacts both gripper sides and starts moving upward.

```bash
python3 scripts/train/train_dual_arm_live.py --arms 4 --n-envs 4
python3 scripts/train/train_dual_arm_live.py --arms 8 --n-envs 2
```

## Features

### Domain Randomisation (`--domain-rand`)

Randomises object mass (0.1–0.6 kg) and surface friction (0.5–3.0) every episode reset. Improves sim-to-real transfer by preventing the policy from overfitting to a single physical configuration.

```bash
python3 scripts/train/train_dual_arm_live.py --arms 4 --n-envs 4 --domain-rand
```

### Auto-Curriculum (`--curriculum auto`)

Self-pacing difficulty scheduler. Starts objects spawned close to the arm (easy grasping), and gradually moves them toward the standard table position as the success rate rises above 70%. Backs off if success rate drops below 30%. Current difficulty is logged in `scene_summary.curriculum_difficulty` each step.

```bash
python3 scripts/train/train_dual_arm_live.py --arms 4 --n-envs 4 --curriculum auto
```

Other curriculum options: `easy_grasp` (fixed easy spawn), `grasp_focus` (object very close to EE, starts in phase 1).

### Grasp Quality Reward

Always active — no flag needed. When both gripper pads contact the object, a symmetry score is computed from the left/right contact force magnitudes. A perfectly centred grasp scores 1.0; a one-sided grasp scores 0.0. The bonus is applied in both the grasp phase and the lift phase, encouraging the policy to maintain a firm, centred grip throughout.

### Handover Mode (`--handover`)

Pairs each left arm with the opposite right arm. Left arms pick their object and carry it to a central handover table (x = 0). Once a left arm places its object there, the paired right arm activates (phase changes from −1 → 0), picks up the same object, and carries it to the right-side drop zone. Both arms must complete for the episode to terminate.

```bash
python3 scripts/train/train_dual_arm_live.py --arms 4 --n-envs 2 --handover
```

Combine flags freely:

```bash
python3 scripts/train/train_dual_arm_live.py \
  --arms 4 --n-envs 4 \
  --curriculum auto \
  --domain-rand \
  --handover \
  --device cuda
```

## Setup

```bash
pip install mujoco gymnasium stable-baselines3
```

Clone [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) to `~/mujoco_menagerie` or set the `MUJOCO_MENAGERIE_PATH` environment variable.

## Project Layout

- `envs/`: MuJoCo Gymnasium environments
- `scripts/train/`: training entrypoints
- `mujoco_ur_rl_ros2/`: ROS2 policy node package
- `launch/`: ROS2 launch files
- `assets/`: README images and visual references

## Train

```bash
# Single arm reach
python3 scripts/train/train.py

# Single arm pick-and-place
python3 scripts/train/train_pick_place.py

# Multi-arm, headless
python3 scripts/train/train_dual_arm_live.py --arms 4 --n-envs 4

# Larger symmetric layout
python3 scripts/train/train_dual_arm_live.py --arms 8 --n-envs 2

# With MuJoCo viewer
python3 scripts/train/train_dual_arm_live.py --arms 4 --n-envs 2 --viewer

# GPU
python3 scripts/train/train_dual_arm_live.py --arms 4 --n-envs 4 --device cuda

# Domain randomisation + auto-curriculum
python3 scripts/train/train_dual_arm_live.py --arms 4 --n-envs 4 --domain-rand --curriculum auto

# Handover task
python3 scripts/train/train_dual_arm_live.py --arms 4 --n-envs 2 --handover

# Shared single-arm policy (every arm contributes a sample per step)
python3 scripts/train/train_shared_arm.py --arms 8 --n-envs 2 --all-arms-samples --device cuda

# Resume a shared-arm run from its latest checkpoint
python3 scripts/train/train_shared_arm.py \
  --arms 8 --n-envs 2 --all-arms-samples --device cuda \
  --resume-model models/shared_arm/shared_arm_8arm_all_samples_resume_20260410_1501/checkpoints/shared_arm_79952_steps.zip \
  --run-name shared_arm_resume_$(date +%Y%m%d_%H%M%S)

# Play back a trained shared-arm policy
python3 scripts/train/play_shared_arm.py \
  --model models/shared_arm/<run_name>/best_model.zip \
  --arms 8 --viewer --deterministic
```

Best multi-arm models and checkpoints are saved under `models/multi_arm/`, run logs under `logs/multi_arm/`.
Shared one-arm policy runs use `models/shared_arm/` and `logs/shared_arm/`.
The best trained `.zip` RL models are committed to the repository. The default ROS/Gazebo nodes automatically read `models/shared_arm/shared_arm_8arm_all_samples_resume_20260410_1501/best_model.zip` and `models/best_model.zip`, so they work out-of-the-box after cloning.

## Monitor Progress

```bash
# Find the newest run directory
cat logs/multi_arm/latest_run.txt

# Watch the live console log
tail -f logs/multi_arm/<run_name>.out

# Check the latest heartbeat
cat logs/multi_arm/<run_name>/latest_status.json
```

Useful signals:
- `timesteps`: training is advancing
- `ep_rew_mean`: average training reward per finished episode
- `eval_mean_reward`: evaluation reward from the latest eval pass
- `fps`: simulation throughput
- `env0_info.scene_summary.curriculum_difficulty`: current auto-curriculum difficulty (0 = easy, 1 = standard)

## ROS2 Package

This repo can also act as a ROS2 Python package named `mujoco_ur_rl_ros2`. The packaged node loads a trained SAC policy and publishes UR5e joint trajectories.

Build it from a ROS2 workspace:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/darshmenon/mujoco-ur-arm-rl.git
cd ~/ros2_ws
colcon build --packages-select mujoco_ur_rl_ros2
source install/setup.bash
```

The ROS2 node still expects the Python environment to have `numpy` and `stable-baselines3` available.

Run the node directly:

```bash
ros2 run mujoco_ur_rl_ros2 ur_policy_node --ros-args \
  -p model_path:=/home/asimov/mujoco-ur-arm-rl/models/best_model.zip \
  -p target_x:=0.4 -p target_y:=0.0 -p target_z:=0.4
```

Or launch it:

```bash
ros2 launch mujoco_ur_rl_ros2 ur_policy.launch.py \
  model_path:=/home/asimov/mujoco-ur-arm-rl/models/best_model.zip
```

Run the shared-arm policy node directly:

```bash
ros2 run mujoco_ur_rl_ros2 shared_arm_policy_node --ros-args \
  -p model_path:=/path/to/mujoco-ur-arm-rl/models/shared_arm/<run_name>/best_model.zip \
  -p arm_trajectory_topic:=/scaled_joint_trajectory_controller/joint_trajectory \
  -p gripper_trajectory_topic:=/gripper_controller/joint_trajectory \
  -p ee_x:=0.0 -p ee_y:=0.0 -p ee_z:=0.0 \
  -p object_x:=-1.18 -p object_y:=0.0 -p object_z:=0.045 \
  -p drop_x:=-1.25 -p drop_y:=0.0 -p drop_z:=0.02
```

Launch Gazebo plus the shared policy (uses the bundled `ur_gazebo` package by default):

```bash
ros2 launch mujoco_ur_rl_ros2 gazebo_shared_arm_policy.launch.py \
  model_path:=/path/to/mujoco-ur-arm-rl/models/shared_arm/<run_name>/best_model.zip \
  ee_x:=0.0 ee_y:=0.0 ee_z:=0.0 \
  object_x:=-1.18 object_y:=0.0 object_z:=0.045 \
  drop_x:=-1.25 drop_y:=0.0 drop_z:=0.02
```

Override `gazebo_launch_path` if you want to use a different Gazebo stack:

```bash
ros2 launch mujoco_ur_rl_ros2 gazebo_shared_arm_policy.launch.py \
  gazebo_launch_path:=/path/to/ur_gazebo/launch/ur.gazebo.launch.py \
  model_path:=/path/to/mujoco-ur-arm-rl/models/shared_arm/<run_name>/best_model.zip
```

Gazebo notes:
- `shared_arm_policy_node` is exported as a ROS2 executable, so `ros2 run` and `ros2 launch` can both find it after rebuilding.
- The shared-arm node defaults to `/scaled_joint_trajectory_controller/joint_trajectory`.
- Override `arm_joint_names` or `gripper_joint_names` if your Gazebo stack uses different joint names.
- The shared policy does not infer object or drop poses from Gazebo; set `ee_*`, `object_*`, `drop_*`, and `phase` to match the scene.

ROS2 files in this repo:
- `package.xml`, `setup.py`, `setup.cfg`
- `mujoco_ur_rl_ros2/ur_policy_node.py`
- `mujoco_ur_rl_ros2/shared_arm_policy_node.py`
- `launch/ur_policy.launch.py`
- `launch/gazebo_shared_arm_policy.launch.py`
- `ros2/ur_policy_node.py` (compatibility wrapper)

## Visualize

```bash
python3 scripts/train/train_dual_arm_live.py --arms 4 --n-envs 1 --viewer
```
