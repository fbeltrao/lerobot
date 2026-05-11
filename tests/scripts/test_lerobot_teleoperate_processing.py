from lerobot.scripts.lerobot_teleoperate import get_processed_teleop_action


class FakeTeleop:
    def get_action(self):
        return {"motor_1.pos": 1.0}


def test_get_processed_teleop_action_uses_existing_processing_order():
    obs = {"motor_1.pos": 10.0}
    calls = []

    def teleop_action_processor(data):
        action, observed = data
        calls.append(("teleop", action, observed))
        return {"motor_1.pos": action["motor_1.pos"] + observed["motor_1.pos"]}

    def robot_action_processor(data):
        action, observed = data
        calls.append(("robot", action, observed))
        return {"motor_1.pos": action["motor_1.pos"] * 2}

    processed = get_processed_teleop_action(
        teleop=FakeTeleop(),
        obs=obs,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
    )

    assert processed == {"motor_1.pos": 22.0}
    assert calls == [
        ("teleop", {"motor_1.pos": 1.0}, obs),
        ("robot", {"motor_1.pos": 11.0}, obs),
    ]
