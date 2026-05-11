import numpy as np
from lerobot_ros.json_codec import decode_action, decode_observation, encode_action, encode_observation


def test_action_round_trip_preserves_feature_keys_and_json_values():
    action = {
        "shoulder_pan.pos": np.float32(12.5),
        "nested": np.array([1, 2, 3]),
    }

    decoded = decode_action(encode_action(action, source_id="teleop", stamp=123.25))

    assert decoded == {
        "source_id": "teleop",
        "stamp": 123.25,
        "action": {"shoulder_pan.pos": 12.5, "nested": [1, 2, 3]},
    }


def test_observation_excludes_image_arrays_from_generic_json():
    observation = {
        "motor_1.pos": np.float64(1.5),
        "front": np.zeros((4, 5, 3), dtype=np.uint8),
        "depth": np.zeros((4, 5), dtype=np.uint8),
    }

    decoded = decode_observation(encode_observation(observation, stamp=456.0, exclude_images=True))

    assert decoded == {"stamp": 456.0, "observation": {"motor_1.pos": 1.5}}


def test_observation_can_include_non_image_arrays():
    observation = {"joint_vector": np.array([1.0, 2.0])}

    decoded = decode_observation(encode_observation(observation, stamp=456.0, exclude_images=True))

    assert decoded["observation"]["joint_vector"] == [1.0, 2.0]
