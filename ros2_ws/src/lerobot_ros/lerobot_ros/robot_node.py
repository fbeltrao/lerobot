from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot.robots.utils import make_robot_from_config

from .config import LeRobotRosConfig, load_config
from .json_codec import decode_action, encode_observation
from .topics import ACTION_TOPIC, CAMERA_TOPIC_TEMPLATE, JOINT_STATES_TOPIC, OBSERVATION_TOPIC

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import String
except ModuleNotFoundError:
    rclpy = None
    Node = object
    QoSProfile = ReliabilityPolicy = qos_profile_sensor_data = None
    Image = JointState = String = None

logger = logging.getLogger(__name__)


@dataclass
class CommandArbitrator:
    timeout_s: float
    active_source_id: str | None = None
    last_command_time_s: float | None = None

    def accept(self, source_id: str, command_time_s: float, now_s: float) -> bool:
        if now_s - command_time_s > self.timeout_s:
            return False
        if self.active_source_id is None or self._active_source_timed_out(now_s):
            self.active_source_id = source_id
            self.last_command_time_s = command_time_s
            return True
        if source_id == self.active_source_id:
            self.last_command_time_s = command_time_s
            return True
        return False

    def _active_source_timed_out(self, now_s: float) -> bool:
        return self.last_command_time_s is None or now_s - self.last_command_time_s > self.timeout_s


class LeRobotRosRobotNode(Node):
    def __init__(self, cfg: LeRobotRosConfig):
        if rclpy is None:
            raise RuntimeError("ROS2 Python packages are required to run lerobot_ros_robot_node")
        if cfg.robot is None:
            raise ValueError("Robot node config must include a 'robot' section")

        super().__init__("lerobot_ros_robot_node")
        self.cfg = cfg
        self.robot = make_robot_from_config(cfg.robot)
        self.robot.connect(calibrate=False)
        if not self.robot.is_connected:
            raise RuntimeError("Robot did not report a connected state after connect(calibrate=False)")
        if not self.robot.is_calibrated:
            raise RuntimeError("Robot is not calibrated. ROS startup assumes existing calibration.")

        self.arbitrator = CommandArbitrator(timeout_s=cfg.command_timeout_s)
        self.action_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.observation_pub = self.create_publisher(String, OBSERVATION_TOPIC, qos_profile_sensor_data)
        self.joint_state_pub = self.create_publisher(JointState, JOINT_STATES_TOPIC, qos_profile_sensor_data)
        self.image_publishers: dict[str, Any] = {}
        self.create_subscription(String, ACTION_TOPIC, self._on_action, self.action_qos)
        self.create_timer(1.0 / cfg.fps, self._publish_observation)

    def _on_action(self, msg: Any) -> None:
        try:
            payload = decode_action(msg)
            now_s = self._now_s()
            source_id = payload["source_id"]
            if not self.arbitrator.accept(source_id, payload["stamp"], now_s):
                self.get_logger().warning("Ignoring stale or competing action source '%s'", source_id)
                return
            action = self._filter_action(payload["action"])
            self.robot.send_action(action)
        except Exception as exc:
            self.get_logger().error("Failed to process ROS LeRobot action: %s", exc)

    def _filter_action(self, action: dict[str, Any]) -> dict[str, Any]:
        action_features = self.robot.action_features
        return {key: value for key, value in action.items() if key in action_features}

    def _publish_observation(self) -> None:
        obs = self.robot.get_observation()
        stamp = self._now_s()
        observation_msg = String()
        observation_msg.data = encode_observation(obs, stamp=stamp, exclude_images=True)
        self.observation_pub.publish(observation_msg)
        self._publish_joint_states(obs)
        self._publish_images(obs)

    def _publish_joint_states(self, obs: dict[str, Any]) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        for key, value in obs.items():
            if _is_scalar(value):
                msg.name.append(key.removesuffix(".pos"))
                msg.position.append(float(value))
        self.joint_state_pub.publish(msg)

    def _publish_images(self, obs: dict[str, Any]) -> None:
        for key, value in obs.items():
            if isinstance(value, np.ndarray) and value.ndim in (2, 3):
                camera_name = _camera_topic_name(key)
                publisher = self.image_publishers.get(camera_name)
                if publisher is None:
                    topic = CAMERA_TOPIC_TEMPLATE.format(camera_name=camera_name)
                    publisher = self.create_publisher(Image, topic, qos_profile_sensor_data)
                    self.image_publishers[camera_name] = publisher
                publisher.publish(_numpy_to_image_msg(value, self.get_clock().now().to_msg()))

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def destroy_node(self) -> bool:
        try:
            if self.robot.is_connected:
                self.robot.disconnect()
        finally:
            destroyed = super().destroy_node()
        return destroyed


def _numpy_to_image_msg(array: np.ndarray, stamp: Any) -> Any:
    image = Image()
    image.header.stamp = stamp
    image.height = int(array.shape[0])
    image.width = int(array.shape[1])
    if array.ndim == 2:
        image.encoding = "mono8"
        channels = 1
    elif array.shape[2] == 3:
        image.encoding = "rgb8"
        channels = 3
    else:
        raise ValueError(f"Unsupported image shape: {array.shape}")
    image.is_bigendian = False
    image.step = image.width * channels
    image.data = np.ascontiguousarray(array.astype(np.uint8, copy=False)).tobytes()
    return image


def _camera_topic_name(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_/]", "_", key).strip("/")


def _is_scalar(value: Any) -> bool:
    return isinstance(value, int | float | np.generic) or (isinstance(value, np.ndarray) and value.ndim == 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args, overrides = parser.parse_known_args()
    cfg = load_config(args.config_path, overrides)
    rclpy.init()
    node = LeRobotRosRobotNode(cfg)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
