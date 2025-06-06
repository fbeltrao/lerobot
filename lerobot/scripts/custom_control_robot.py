
# Inspired by https://huggingface.co/yinchenghust/openpi_base

import time
import torch
from lerobot.common.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.robot_devices.robots.utils import make_robot
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature

###################################################
# Requires packages:
# - pip install transformers==4.48.1
#
###################################################

robot = make_robot("so101", leader_arms={})
robot.connect()
obs = robot.capture_observation()
print("Observation keys:", obs.keys())
print("Robot ready!")

OBS_STATE_KEY = "observation.state"
OBS_IMAGE_FRONT_KEY = "observation.images.front"
OBS_IMAGE_WRIST_KEY = "observation.images.wrist"

# policy = PI0FASTPolicy.from_pretrained("Xiaoyan97/so100-pi0fast")
pretrained_name_or_path = "lerobot/pi0"
pretrained_name_or_path = "Xiaoyan97/so100-pi0fast"
pretrained_name_or_path = "sengi/pi0_so100_pretrain_500"
pretrained_name_or_path = "fbeltrao/pi0fast_so101_unplug_cable_10_stepslast"
policy_config = PreTrainedConfig.from_pretrained(pretrained_name_or_path=pretrained_name_or_path)
policy_config.device="cpu"  # Force CPU for compatibility

# Inspired from https://huggingface.co/datasets/yinchenghust/libero_rich_lang_all/blob/main/meta/info.json
# More info: https://github.com/huggingface/lerobot/issues/694

# Reset features
policy_config.input_features = {}
policy_config.output_features = {}

policy_config.input_features.update({ OBS_IMAGE_FRONT_KEY: PolicyFeature(FeatureType.VISUAL, shape=(480, 640, 3)) })
policy_config.input_features.update({ OBS_IMAGE_WRIST_KEY: PolicyFeature(FeatureType.VISUAL, shape=(480, 640, 3)) })
policy_config.output_features.update({ "action": PolicyFeature(FeatureType.ACTION, shape=(6,)) })  # Adjust shape as needed

policy = PI0FASTPolicy.from_pretrained(pretrained_name_or_path, config=policy_config)
policy.eval()
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"  # Force CPU for compatibility
policy.to(device)

prompt = "Pick up the pen."
prompt = "Unplug the cable."

def obs_to_image(img: torch.Tensor) -> torch.Tensor:
    v = img.type(torch.float32) / 255
    result = v.permute(2, 0, 1).unsqueeze(0)
    return result


while True:
    start_time = time.perf_counter()
    print("Capturing observation...")
    obs = robot.capture_observation()
    obs[OBS_IMAGE_FRONT_KEY] = obs_to_image(obs[OBS_IMAGE_FRONT_KEY])
    obs[OBS_IMAGE_WRIST_KEY] = obs_to_image(obs[OBS_IMAGE_WRIST_KEY])
    obs[OBS_STATE_KEY] = obs[OBS_STATE_KEY].unsqueeze(0)
    obs["task"] = [prompt]

    with torch.no_grad():
        print("Running policy...")
        action = policy.select_action(obs)
        action  = action.squeeze(0).cpu().numpy()
    
    print(f"Sending action: {action}")
    robot.send_action(torch.tensor(action))
    dt = time.perf_counter() - start_time
    wait_seconds = 1 / 30 - dt
    print(f"Will wait {wait_seconds:.4f} seconds")
    busy_wait(wait_seconds)
    print(f"Frame processed in {dt:.4f} seconds")