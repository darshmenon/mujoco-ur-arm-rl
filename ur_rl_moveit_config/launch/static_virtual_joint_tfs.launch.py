from ur_rl_moveit_configs_utils import MoveItConfigsBuilder
from ur_rl_moveit_configs_utils.launches import generate_static_virtual_joint_tfs_launch


def generate_launch_description():
    ur_rl_moveit_config = MoveItConfigsBuilder("ur", package_name="ur_rl_moveit_config").to_ur_rl_moveit_configs()
    return generate_static_virtual_joint_tfs_launch(ur_rl_moveit_config)
