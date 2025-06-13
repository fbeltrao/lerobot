 
# Inspired by https://huggingface.co/yinchenghust/openpi_base

from dataclasses import dataclass
import time
from typing import Any
import draccus
import requests
import torch
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig # DO NOT REMOVE: required for the config parser
from lerobot.common.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.common.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy

from lerobot.common.utils.robot_utils import busy_wait
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature

from lerobot.common.cameras import (  # noqa: F401
   CameraConfig,
    )


from lerobot.common.cameras.opencv import (  # noqa: F401
   OpenCVCameraConfig,
    )


from lerobot.common.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
    so101_follower
    )


from lerobot.common.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    so101_leader
)


import base64
import numpy as np
import io
from PIL import Image


###################################################
# Requires packages:
# - pip install transformers==4.48.1
#
###################################################



OBS_STATE_KEY = "observation.state"
OBS_IMAGE_FRONT_KEY = "observation.images.front"
OBS_IMAGE_WRIST_KEY = "observation.images.wrist"



def connect_robot(config: RobotConfig) -> Robot:
    
    robot = make_robot_from_config(config)
    robot.connect()
    obs = robot.get_observation()
    print("Observation keys:", obs.keys())
    print("Robot ready!")
    return robot


def create_local_policy() -> PI0FASTPolicy:

    # policy = PI0FASTPolicy.from_pretrained("Xiaoyan97/so100-pi0fast")
    pretrained_name_or_path = "lerobot/pi0"
    pretrained_name_or_path = "Xiaoyan97/so100-pi0fast"
    pretrained_name_or_path = "sengi/pi0_so100_pretrain_500"
    pretrained_name_or_path = "fbeltrao/pi0fast_so101_unplug_cable_10_stepslast"
    pretrained_name_or_path = "fbeltrao/pi0fast_so101_unplug_cable_5000_steps"
    policy_config = PreTrainedConfig.from_pretrained(pretrained_name_or_path=pretrained_name_or_path)
    policy_config.device="cpu"  # Force CPU for compatibility
    # policy_config.device="cuda"  # Force CPU for compatibility

    # Inspired from https://huggingface.co/datasets/yinchenghust/libero_rich_lang_all/blob/main/meta/info.json
    # More info: https://github.com/huggingface/lerobot/issues/694

    # # Reset features
    # policy_config.input_features = {}
    # policy_config.output_features = {}

    # policy_config.input_features.update({ OBS_IMAGE_FRONT_KEY: PolicyFeature(FeatureType.VISUAL, shape=(480, 640, 3)) })
    # policy_config.input_features.update({ OBS_IMAGE_WRIST_KEY: PolicyFeature(FeatureType.VISUAL, shape=(480, 640, 3)) })
    # policy_config.output_features.update({ "action": PolicyFeature(FeatureType.ACTION, shape=(6,)) })  # Adjust shape as needed

    policy = PI0FASTPolicy.from_pretrained(pretrained_name_or_path, config=policy_config)
    policy.eval()
    # # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"  # Force CPU for compatibility
    policy.to(device)
    return policy




def obs_to_image(img: torch.Tensor) -> torch.Tensor:
    v = img.type(torch.float32) / 255
    result = v.permute(2, 0, 1).unsqueeze(0)
    return result


from requests.adapters import HTTPAdapter, Retry

class PolicyClient:
    def __init__(self, api_url: str, robot: Robot):
        self.api_url = api_url
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504],)
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        self.robot = robot
        self.obs_features = hw_to_dataset_features(robot.observation_features, "observation")
   
    def image_to_base64(self, img: np.ndarray) -> str:
        # img: np.ndarray with shape (480, 640, 3), dtype=np.uint8
        pil_img = Image.fromarray(img, "RGB")
        with io.BytesIO() as buffer:
            pil_img.save(buffer, format="PNG")
            base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return base64_str

    def select_action(self, obs: dict[str, np.ndarray]) -> list[list[float]]:
        predict_request = { "images": {}, **obs}
        for k, v in obs.items():
            if "image" in k:            
                predict_request["images"][k] = self.image_to_base64(v)
                predict_request.pop(k)
            elif isinstance(v, np.ndarray):
                predict_request[k] = v.tolist()        

        predict_request["state"] = predict_request.pop(OBS_STATE_KEY, None)
        predict_request["task"] = obs.get("task", "")
        
        try:
            res = self.session.post(f"{self.api_url}/predict", json=predict_request, timeout=10)
            if res.status_code != 200:
                print(f"Error: {res.status_code} - {res.text}")
                return []
        except requests.Timeout:
            print("Request timed out...")
            return []
            
        return res.json()["actions"]



def move_to_start_position(robot: Robot):
    # Move the robot to a predefined start position
    # This is a placeholder function, you should implement the actual logic to move the robot
    print("Moving robot to start position...")
    start_position = [10.0,82.0,75.0,83.0,3.0,-0.0]
    start_position = [2.64,193.62,172.18,79.54,-6.33,0.14]
    start_position = [-1.20,-99.23,98.30,74.79,-1.16,2.83]
    # robot.send_action(torch.tensor(start_position, dtype=torch.float32, device="cpu"))  # Example action, adjust as needed
    # busy_wait(2)
    print("Robot moved to start position.")



@dataclass
class CustomControlConfig:
    robot: RobotConfig
    teleop: TeleoperatorConfig | None = None

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["robot"]


@draccus.wrap()
def remote_inference_control(cfg: CustomControlConfig, prompt: str):
    robot = connect_robot(cfg.robot)
    move_to_start_position(robot)

    policy = PolicyClient("http://192.168.0.127:8000", robot)  # Adjust the URL to your API endpoint
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")

    while True:

        print("Capturing observation...")
        obs = robot.get_observation()
        observation_frame = build_dataset_frame(obs_features, obs, prefix="observation")

        observation_frame["task"] = prompt
        actions = policy.select_action(observation_frame)

        for action in actions:
            start_time = time.perf_counter()
            print(f"Sending action: {action}") 
            robot_action = {key: action[i] for i, key in enumerate(robot.action_features)}
            robot.send_action(robot_action)
            dt = time.perf_counter() - start_time
            wait_seconds = 1 / 30 - dt
            print(f"Will wait {wait_seconds:.4f} seconds")
            busy_wait(wait_seconds)
            print(f"Frame processed in {dt:.4f} seconds")
        busy_wait(1)


def generate_local_inference_obs(obs: dict[str, Any], prompt: str) -> dict[str, Any]:
    return {
        OBS_IMAGE_FRONT_KEY: obs_to_image(obs[OBS_IMAGE_FRONT_KEY]),
        OBS_IMAGE_WRIST_KEY: obs_to_image(obs[OBS_IMAGE_WRIST_KEY]),
        OBS_STATE_KEY: obs[OBS_STATE_KEY].unsqueeze(0),
        "task": [prompt]
    }

def local_inference_control(prompt: str):
    robot = connect_robot()
    move_to_start_position(robot)
    policy = create_local_policy()

    while True:
        start_time = time.perf_counter()
        print("Capturing observation...")
        obs = robot.get_observation()
        obs_for_policy = generate_local_inference_obs(obs, prompt)

        with torch.no_grad():
            print(f"Running policy with state {obs_for_policy[OBS_STATE_KEY]}...")
            action = policy.select_action(obs_for_policy)
            action  = action.squeeze(0).cpu().numpy()
        
        print(f"Sending action: {action}")
        robot.send_action(torch.tensor(action))
        dt = time.perf_counter() - start_time
        wait_seconds = 1 / 30 - dt
        print(f"Will wait {wait_seconds:.4f} seconds")
        busy_wait(wait_seconds)
        print(f"Frame processed in {dt:.4f} seconds")

if __name__ == "__main__":
    print("Starting local inference control loop...")

    prompt = "Pick up the pen."
    prompt = "Unplug the cable."
    prompt = "Pick up the lego brick."
    # local_inference_control(prompt)
    remote_inference_control(prompt)
    print("Control loop finished.")