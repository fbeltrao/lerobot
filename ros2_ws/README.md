# LeRobot ROS2 Workspace

This workspace contains `lerobot_ros`, a ROS2 bridge package that runs a LeRobot robot as the ROS-owned hardware process and lets teleoperation publish LeRobot-native actions over ROS topics.

The bridge keeps the robot node as the only owner of the physical robot. The teleop node owns only the teleoperator, subscribes to observations, processes actions through the same LeRobot teleop path, and publishes actions to the robot node.

## Prerequisites

- ROS2 with Python support sourced in your shell.
- LeRobot installed in the Python environment used by ROS2.
- Existing LeRobot robot calibration. The robot node connects with calibration disabled and fails if the robot is not already usable.

From the repository root, install LeRobot dependencies if needed:

```bash
uv sync --locked
```

Then source ROS2 and make sure this repository is importable by the ROS2 Python environment you will use for `colcon build` and `ros2 run`.

## Build

From this directory:

```bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

If LeRobot is not installed into the same environment, install it editable from the repository root before building:

```bash
cd ..
uv pip install -e .
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## Configuration

Both nodes load a small wrapper config with `--config_path`. The robot node requires a `robot` section. The teleop node requires a `teleop` section.

Example robot config:

```yaml
robot:
  type: so101_follower
  id: my_robot
  port: /dev/ttyACM0
  cameras:
    front:
      type: opencv
      index_or_path: 0
      width: 640
      height: 480
      fps: 30
fps: 30
command_timeout_s: 0.5
```

Example teleop config:

```yaml
teleop:
  type: so101_leader
  id: my_leader
  port: /dev/ttyACM1
fps: 30
source_id: teleop
```

Use the same robot and teleoperator config fields you would pass to normal LeRobot CLI tools. The wrapper fields are:

- `fps`: publish or control-loop rate.
- `command_timeout_s`: how long the active command source keeps control without sending a fresh command.
- `source_id`: action source name published by the teleop node.

## Run Both Nodes

After building and sourcing the workspace:

```bash
ros2 launch lerobot_ros teleop.launch.py \
  robot_config_path:=/path/to/robot.yaml \
  teleop_config_path:=/path/to/teleop.yaml \
  fps:=30 \
  command_timeout_s:=0.5
```

## Run Nodes Separately

Robot owner process:

```bash
ros2 run lerobot_ros lerobot_ros_robot_node --config_path /path/to/robot.yaml
```

Teleop adapter:

```bash
ros2 run lerobot_ros lerobot_ros_teleop --config_path /path/to/teleop.yaml
```

You can override wrapper fields through CLI arguments after `--config_path`:

```bash
ros2 run lerobot_ros lerobot_ros_robot_node --config_path /path/to/robot.yaml --fps 60 --command_timeout_s 0.25
```

## Topics

Robot node publishes:

- `/joint_states`
- `/lerobot/observation`
- `/lerobot/cameras/<camera_name>/image_raw`

Robot node subscribes:

- `/lerobot/action`

Teleop node subscribes:

- `/lerobot/observation`

Teleop node publishes:

- `/lerobot/action`

The generic LeRobot topics use `std_msgs/msg/String` containing JSON. Feature keys are preserved exactly, for example `shoulder_pan.pos`. Image pixel data is not included in `/lerobot/observation`; camera frames are published as `sensor_msgs/msg/Image`.

## Smoke Checks

In another sourced terminal, inspect the robot publications:

```bash
ros2 topic list
ros2 topic echo /joint_states
ros2 topic echo /lerobot/observation
```

For cameras:

```bash
ros2 topic list | grep /lerobot/cameras
ros2 topic echo /lerobot/cameras/front/image_raw --once
```

To verify actions are flowing from teleop:

```bash
ros2 topic echo /lerobot/action
```

If a second action source publishes while teleop is active, the robot node ignores it until the active source times out.

## Tests

From the repository root:

```bash
uv run pytest ros2_ws/src/lerobot_ros/test tests/scripts/test_lerobot_teleoperate_processing.py -q
```

These tests cover JSON encoding, image exclusion from generic observations, command-source arbitration, and the shared teleop action-processing helper. Hardware validation is a manual smoke path because it depends on a calibrated robot and local ROS2 setup.