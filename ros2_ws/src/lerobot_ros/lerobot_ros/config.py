from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import draccus

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    omx_follower,
    openarm_follower,
    reachy2,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_so_leader,
    gamepad,
    homunculus,
    keyboard,
    koch_leader,
    omx_leader,
    openarm_leader,
    openarm_mini,
    reachy2_teleoperator,
    so_leader,
    unitree_g1,
)
from lerobot.utils.import_utils import register_third_party_plugins


@dataclass
class LeRobotRosConfig:
    robot: RobotConfig | None = None
    teleop: TeleoperatorConfig | None = None
    fps: int = 30
    command_timeout_s: float = 0.5
    source_id: str = "teleop"


def load_config(config_path: str | Path, args: Sequence[str] | None = None) -> LeRobotRosConfig:
    register_third_party_plugins()
    return draccus.parse(
        config_class=LeRobotRosConfig,
        config_path=Path(config_path),
        args=[] if args is None else list(args),
    )
