## Make a web API to collect observations and execute actions
# List of dependencies
# - fastapi
# - uvicorn
# - transformers
# - pydantic
# - pillow
# pip install fastapi uvicorn transformers==4.48.1 pydantic pillow
import base64
from dataclasses import dataclass
import io
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.control_utils import predict_action
from lerobot.common.utils.utils import get_safe_torch_device
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig

# Initialize FastAPI app
app = FastAPI(
    title="Lerobot policy Model API",
    description="API for interacting with the HuggingFace Lerobot policies",
    version="1.0.0",
)


# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ObservationRequest(BaseModel):
    """Request model for sending both image and prompt"""

    images: dict[str, str]
    state: list[float]
    task: str
    robot_type: str | None = None


class PolicyResponse(BaseModel):
    """Response model with text and actions"""

    actions: list[list[float]]


policy: PreTrainedPolicy | None = None
processor: Callable[[ObservationRequest], list[torch.Tensor]] | None = None


def tensor_from_image_base64(name: str, image_base64: str, *, store_image: bool = False) -> torch.Tensor:
    """Convert a base64 encoded image to a PyTorch tensor"""
    return torch.from_numpy(nparray_from_image_base64(name, image_base64, store_image=store_image))

def nparray_from_image_base64(name: str, image_base64: str, *, store_image: bool = False) -> np.ndarray:
    """Convert a base64 encoded image to a PyTorch tensor"""
    with io.BytesIO(base64.b64decode(image_base64)) as buffer:
        pil_img = Image.open(buffer).convert("RGB")
        if store_image:
            debug_folder = Path("outputs") / "predictions" / "input"
            debug_folder.mkdir(parents=True, exist_ok=True)
            pil_img.save(debug_folder / f"{name}.png")  # Save the image for debugging
        return np.array(pil_img)


OBS_STATE_KEY = "observation.state"


def process_policy(request: ObservationRequest) -> list[list[float]]:
    """Process the observation request to extract images and state"""
    global policy
    if policy is None:
        raise HTTPException(status_code=503, detail="Policy model not loaded yet")

    obs = {
        OBS_STATE_KEY: np.array(request.state, dtype=np.float32),
    }

    for img_name, image_value in request.images.items():
        obs[img_name] = nparray_from_image_base64(img_name, image_value, store_image=True)

    predicted = predict_action(
        observation=obs, 
        policy=policy, 
        device=get_safe_torch_device(policy.config.device), 
        use_amp=policy.config.use_amp,
        task=request.task or "",
        robot_type=request.robot_type or "",
    )

    # Convert the predicted actions to a list of lists
    actions = []
    actions.append(predicted.cpu().numpy().tolist())

    if "_action_queue" in policy.__dict__:
        # If the policy has an action queue, we need to pop the first action
        while len(policy._action_queue) > 0:
            action = policy._action_queue.popleft()
            actions.append(action.cpu().numpy()[0].tolist())

    return actions


@app.on_event("startup")
async def on_startup():
    """Event handler for application startup"""
    global policy, processor
    policy, processor = await load_model()

async def load_model() -> tuple[PreTrainedPolicy, Callable[[ObservationRequest], list[torch.Tensor]]] :
    """Load the model when the application starts"""
    
    # pretrained_name_or_path = "fbeltrao/pi0fast_so101_multi_task"
    # revision = "v5000steps"

    # print(f"Loading policy model {pretrained_name_or_path} (revision={revision})... This may take a while.")

    # policy = PI0FASTPolicy.from_pretrained(pretrained_name_or_path, revision=revision)
    

    pretrained_name_or_path = "fbeltrao/act_so101_multi_task"
    revision= "v5000steps"
    policy = ACTPolicy.from_pretrained(pretrained_name_or_path, revision=revision)

    policy.eval()
    policy.reset()

    return policy, process_policy


@app.post("/predict", response_model=PolicyResponse)
async def predict(request: ObservationRequest) -> PolicyResponse:
    """Endpoint to make predictions using a LeRobot policy"""
    if policy is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        prediction = process_policy(request)
        return PolicyResponse(actions=prediction)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.get("/health")
async def health_check():
    """Endpoint to check if the API is running"""
    return {"status": "healthy", "model_loaded": policy is not None}

if __name__ == "__main__":
    # Run the server on 0.0.0.0 to accept connections from any IP address
    uvicorn.run(app, host="0.0.0.0", port=8000)
