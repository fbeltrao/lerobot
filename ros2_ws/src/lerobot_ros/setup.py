from glob import glob
from pathlib import Path

from setuptools import find_packages, setup

package_name = "lerobot_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (str(Path("share") / package_name / "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "lerobot"],
    zip_safe=True,
    maintainer="LeRobot",
    maintainer_email="lerobot@huggingface.co",
    description="ROS2 bridge package for controlling LeRobot robots through ROS topics.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "lerobot_ros_robot_node = lerobot_ros.robot_node:main",
            "lerobot_ros_teleop = lerobot_ros.teleop_node:main",
        ],
    },
)
