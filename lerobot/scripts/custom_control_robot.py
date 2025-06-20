 
# Inspired by https://huggingface.co/yinchenghust/openpi_base

import base64
import io
import threading
import time
from dataclasses import dataclass

import draccus
import numpy as np
import requests
import torch
from PIL import Image
from requests.adapters import HTTPAdapter, Retry

from lerobot.common.cameras import (  # DO NOT REMOVE: required for the config parser  # noqa: F401, F811
    CameraConfig,
)
from lerobot.common.cameras.opencv import (  # DO NOT REMOVE: required for the config parser  # noqa: F401, F811
    OpenCVCameraConfig,
)
from lerobot.common.cameras.opencv.configuration_opencv import (
    OpenCVCameraConfig,  # DO NOT REMOVE: required for the config parser  # noqa: F401, F811
)
from lerobot.common.constants import OBS_STATE
from lerobot.common.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
from lerobot.common.robots import (  # DO NOT REMOVE: required for the config parser  # noqa: F401, F811
    Robot,
    RobotConfig,
    make_robot_from_config,
    so101_follower,
)
from lerobot.common.teleoperators import (  # DO NOT REMOVE: required for the config parser  # noqa: F401, F811
    TeleoperatorConfig,
    so101_leader,
)
from lerobot.common.utils.control_utils import predict_action
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device
from lerobot.configs.policies import PreTrainedConfig

###################################################
# Requires packages:
# - pip install transformers==4.48.1
#
###################################################


def connect_robot(config: RobotConfig) -> Robot:
    
    robot = make_robot_from_config(config)
    robot.connect()
    obs = robot.get_observation()
    print("Observation keys:", obs.keys())
    print("Robot ready!")
    return robot


def create_local_policy() -> PI0FASTPolicy:

    pretrained_name_or_path = "fbeltrao/pi0fast_so101_multi_task"
    pretrained_name_or_path = "fbeltrao/act_so101_multi_task"
    
    revision = "v5000steps"
    policy_config = PreTrainedConfig.from_pretrained(pretrained_name_or_path=pretrained_name_or_path, revision=revision)
    policy_config.device="cpu"  # Force CPU for compatibility
    
    # policy = PI0FASTPolicy.from_pretrained(pretrained_name_or_path, config=policy_config, revision=revision)
    policy = ACTPolicy.from_pretrained(pretrained_name_or_path, config=policy_config, revision=revision)

    policy.eval()
    device = "cpu"  # Force CPU for compatibility
    policy.to(device)
    return policy


def obs_to_image(img: torch.Tensor) -> torch.Tensor:
    v = img.type(torch.float32) / 255
    result = v.permute(2, 0, 1).unsqueeze(0)
    return result


class PolicyClient:
    def __init__(self, api_url: str, robot: Robot):
        self.api_url = api_url
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504],)
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        self.robot = robot
        self.obs_features = hw_to_dataset_features(robot.observation_features, "observation")
        self.robot_type = robot.robot_type
   
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

        predict_request["state"] = predict_request.pop(OBS_STATE, None)
        predict_request["task"] = obs.get("task", "")
        predict_request["robot_type"] = self.robot_type or ""
        
        try:
            res = self.session.post(f"{self.api_url}/predict", json=predict_request, timeout=10)
            if res.status_code != 200:
                print(f"Error: {res.status_code} - {res.text}")
                return []
        except requests.Timeout:
            print("Request timed out...")
            return []
            
        return res.json()["actions"]


class PromptHandler:
    def __init__(self, initial_prompt: str = ""):
        self.current_prompt = initial_prompt
        self.lock = threading.Lock()
        self.input_thread = None
        self.stop_flag = False
        
    def start(self):
        """Start the background thread to handle user input"""
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()
        
    def stop(self):
        """Stop the input handler"""
        self.stop_flag = True
        
    def get_prompt(self) -> str:
        """Get the current prompt in a thread-safe way"""
        with self.lock:
            return self.current_prompt
            
    def _input_loop(self):
        """Background loop to handle user input"""
        while not self.stop_flag:
            try:
                print("\n" + "="*50)
                print("Current prompt:", f"'{self.get_prompt()}'")
                print("Enter new prompt (or press Enter to keep current, 'quit' to stop):")
                user_input = input("> ").strip()
                
                if user_input.lower() == 'quit':
                    with self.lock:
                        self.current_prompt = ""
                    break
                elif user_input:  # Non-empty input
                    with self.lock:
                        self.current_prompt = user_input
                    print(f"Updated prompt to: '{user_input}'")
                    
            except (EOFError, KeyboardInterrupt):
                with self.lock:
                    self.current_prompt = ""
                break



def move_to_start_position(robot: Robot):
    # Move the robot to a predefined start position
    # This is a placeholder function, you should implement the actual logic to move the robot
    print("Moving robot to start position...")
    # Uncomment and modify one of these start positions as needed:
    # start_position = [10.0,82.0,75.0,83.0,3.0,-0.0]
    # start_position = [2.64,193.62,172.18,79.54,-6.33,0.14]
    # start_position = [-1.20,-99.23,98.30,74.79,-1.16,2.83]
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
    print("Starting remote inference control loop...")
    robot = connect_robot(cfg.robot)
    move_to_start_position(robot)

    policy = PolicyClient("http://192.168.0.127:8000", robot)  # Adjust the URL to your API endpoint
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    
    # Initialize the prompt handler with the initial prompt
    prompt_handler = PromptHandler(prompt)
    prompt_handler.start()
    
    print(f"Starting with prompt: '{prompt}'")
    print("You can change the prompt at any time by typing in the console.")

    try:
        while True:
            # Get the current prompt
            current_prompt = prompt_handler.get_prompt()
            
            # Exit if prompt is empty (user typed 'quit' or interrupted)
            if not current_prompt:
                print("Empty prompt received. Stopping control loop...")
                break

            # print("Capturing observation...")
            obs = robot.get_observation()
            # print only the positions
            # robot_position = { k: v for k, v in obs.items() if k.endswith(".pos") }
            # print(f"Robot position: {robot_position}")
            observation_frame = build_dataset_frame(obs_features, obs, prefix="observation")

            observation_frame["task"] = current_prompt
            actions = policy.select_action(observation_frame)

            for action in actions:
                start_time = time.perf_counter()
                robot_action = {key: action[i] for i, key in enumerate(robot.action_features)}               
                # print(f"Sending action to robot: {robot_action}")
                robot.send_action(robot_action)
                dt = time.perf_counter() - start_time
                # wait_seconds = max(1.0, 1 / 30 - dt)
                wait_seconds = 1 / 30 - dt
                # print(f"Waiting {wait_seconds:.4f} seconds...")
                busy_wait(wait_seconds)            
            busy_wait(1)
    
    except KeyboardInterrupt:
        print("\nControl loop interrupted by user.")
    finally:
        prompt_handler.stop()
        print("Control loop finished.")

@draccus.wrap()
def local_inference_control(cfg: CustomControlConfig, prompt: str):
    print("Starting local inference control loop...")
    robot = connect_robot(cfg.robot)
    move_to_start_position(robot)
    policy = create_local_policy()
    torch_device = get_safe_torch_device(policy.config.device)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    while True:
        start_time = time.perf_counter()

        action_values: torch.Tensor

        # Prevent getting observations if we already have the next action in queue
        if "_action_queue" in policy.__dict__ and len(policy._action_queue) > 0:
            print("Skipping observation capture, action already in queue...")
            action_values = policy._action_queue.popleft().squeeze(0).to("cpu")
        else:

            print("Capturing observation...")
            obs = robot.get_observation()

            # print only the positions
            robot_position = { k: v for k, v in obs.items() if k.endswith(".pos") }
            print(f"Robot position: {robot_position}")

            dataset_frame = build_dataset_frame(obs_features, obs, prefix="observation")
                       
            action_values = predict_action(
                observation=dataset_frame, 
                policy=policy, 
                device=torch_device, 
                use_amp=policy.config.use_amp,
                task=prompt,
                robot_type=robot.robot_type,
            )

        action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}
            
        print(f"Sending action to robot: {action}")
        robot.send_action(action)

        dt = time.perf_counter() - start_time
        wait_seconds = 1 / 30 - dt
        print(f"Waiting {wait_seconds:.4f} seconds...")
        busy_wait(wait_seconds)


if __name__ == "__main__":    
    # Set initial prompt - this can be changed during runtime via console input
    initial_prompt = "Pick up the pen."
    # initial_prompt = "Unplug the cable."
    # initial_prompt = "Pick up the lego brick."
    
    print("Starting robot control with interactive prompts...")
    print("You can change the task prompt at any time during execution.")
    print("Type 'quit' when prompted to stop the robot.")
    
    #local_inference_control(initial_prompt)
    remote_inference_control(initial_prompt)