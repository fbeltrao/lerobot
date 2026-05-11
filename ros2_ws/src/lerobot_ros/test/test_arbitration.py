from lerobot_ros.robot_node import CommandArbitrator


def test_arbitration_keeps_first_fresh_source_until_timeout():
    arbitrator = CommandArbitrator(timeout_s=0.5)

    assert arbitrator.accept("teleop", command_time_s=1.0, now_s=1.0)
    assert arbitrator.accept("teleop", command_time_s=1.2, now_s=1.2)
    assert not arbitrator.accept("external", command_time_s=1.3, now_s=1.3)

    assert arbitrator.accept("external", command_time_s=1.8, now_s=1.8)
    assert arbitrator.active_source_id == "external"


def test_arbitration_rejects_stale_commands():
    arbitrator = CommandArbitrator(timeout_s=0.5)

    assert not arbitrator.accept("teleop", command_time_s=1.0, now_s=1.6)
    assert arbitrator.active_source_id is None
