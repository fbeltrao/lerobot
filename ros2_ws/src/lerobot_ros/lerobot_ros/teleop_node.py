from __future__ import annotations

import argparse
from typing import Any

from lerobot.processor import make_default_processors
from lerobot.scripts.lerobot_teleoperate import get_processed_teleop_action
from lerobot.teleoperators.utils import make_teleoperator_from_config

from .config import LeRobotRosConfig, load_config
from .json_codec import decode_observation, encode_action
from .topics import ACTION_TOPIC, OBSERVATION_TOPIC

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from std_msgs.msg import String
except ModuleNotFoundError:
    rclpy = None
    Node = object
    QoSProfile = ReliabilityPolicy = qos_profile_sensor_data = None
    String = None


class LeRobotRosTeleopNode(Node):
    def __init__(self, cfg: LeRobotRosConfig):
        if rclpy is None:
            raise RuntimeError("ROS2 Python packages are required to run lerobot_ros_teleop")
        if cfg.teleop is None:
            raise ValueError("Teleop node config must include a 'teleop' section")

        super().__init__("lerobot_ros_teleop")
        self.cfg = cfg
        self.teleop = make_teleoperator_from_config(cfg.teleop)
        self.teleop.connect()
        self.latest_observation: dict[str, Any] | None = None
        self.teleop_action_processor, self.robot_action_processor, _ = make_default_processors()
        self.action_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.action_pub = self.create_publisher(String, ACTION_TOPIC, self.action_qos)
        self.create_subscription(String, OBSERVATION_TOPIC, self._on_observation, qos_profile_sensor_data)
        self.create_timer(1.0 / cfg.fps, self._publish_action)

    def _on_observation(self, msg: Any) -> None:
        self.latest_observation = decode_observation(msg)["observation"]

    def _publish_action(self) -> None:
        if self.latest_observation is None:
            return
        action = get_processed_teleop_action(
            teleop=self.teleop,
            obs=self.latest_observation,
            teleop_action_processor=self.teleop_action_processor,
            robot_action_processor=self.robot_action_processor,
        )
        msg = String()
        msg.data = encode_action(action, source_id=self.cfg.source_id, stamp=self._now_s())
        self.action_pub.publish(msg)

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def destroy_node(self) -> bool:
        try:
            if self.teleop.is_connected:
                self.teleop.disconnect()
        finally:
            destroyed = super().destroy_node()
        return destroyed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args, overrides = parser.parse_known_args()
    cfg = load_config(args.config_path, overrides)
    rclpy.init()
    node = LeRobotRosTeleopNode(cfg)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
