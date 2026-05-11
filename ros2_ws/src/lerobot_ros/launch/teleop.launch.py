from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    robot_config_path = LaunchConfiguration("robot_config_path")
    teleop_config_path = LaunchConfiguration("teleop_config_path")
    fps = LaunchConfiguration("fps")
    command_timeout_s = LaunchConfiguration("command_timeout_s")

    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_config_path"),
            DeclareLaunchArgument("teleop_config_path"),
            DeclareLaunchArgument("fps", default_value="30"),
            DeclareLaunchArgument("command_timeout_s", default_value="0.5"),
            Node(
                package="lerobot_ros",
                executable="lerobot_ros_robot_node",
                name="lerobot_ros_robot_node",
                output="screen",
                arguments=[
                    "--config_path",
                    robot_config_path,
                    "--fps",
                    fps,
                    "--command_timeout_s",
                    command_timeout_s,
                ],
            ),
            Node(
                package="lerobot_ros",
                executable="lerobot_ros_teleop",
                name="lerobot_ros_teleop",
                output="screen",
                arguments=["--config_path", teleop_config_path, "--fps", fps],
            ),
        ]
    )
