import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mujoco_ur_rl_ros2.ur_policy_node import main


if __name__ == "__main__":
    main()
