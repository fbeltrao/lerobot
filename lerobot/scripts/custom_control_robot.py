
# Inspired by https://huggingface.co/yinchenghust/openpi_base

import time
from typing import Any
import requests
import torch
from lerobot.common.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.robot_devices.robots.utils import Robot, make_robot
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
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



def connect_robot() -> Robot:
    
    robot = make_robot("so101", leader_arms={})
    robot.connect()
    obs = robot.capture_observation()
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
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504],)
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

    def select_action(self, obs: dict[str, torch.Tensor | str]) -> list[list[float]]:

        
        def image_to_base64(img: np.ndarray) -> str:
            # img: np.ndarray with shape (480, 640, 3), dtype=np.uint8
            pil_img = Image.fromarray(img.numpy(), "RGB")
            with io.BytesIO() as buffer:
                pil_img.save(buffer, format="PNG")
                base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
                return base64_str

        images = {k: image_to_base64(v) for k, v in obs.items() if ".images" in k}
        predict_request = {
            "state": obs[OBS_STATE_KEY].numpy().tolist(),
            "images": images,
            "task": obs.get("task", [""]),
        }
        
        
        try:
            res = self.session.post(f"{self.api_url}/predict", json=predict_request, timeout=10)
            if res.status_code != 200:
                print(f"Error: {res.status_code} - {res.text}")
                return []
        except requests.Timeout:
            print("Request timed out...")
            return []
            
        return res.json()["actions"]





def remote_inference_control(prompt: str):
    robot = connect_robot()

    policy = PolicyClient("http://localhost:8000")  # Adjust the URL to your API endpoint

    while True:

        print("Capturing observation...")
        obs = robot.capture_observation()

        print("Local observation:", generate_local_inference_obs(obs, prompt))

        obs["task"] = [prompt]

        actions = policy.select_action(obs)

        for action in actions:
            start_time = time.perf_counter()
            print(f"Sending action: {action}")
            robot.send_action(torch.tensor(action))
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

    policy = create_local_policy()

    while True:
        start_time = time.perf_counter()
        print("Capturing observation...")
        obs = robot.capture_observation()
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
    # local_inference_control(prompt)
    remote_inference_control(prompt)
    print("Control loop finished.")