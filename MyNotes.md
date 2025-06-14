# Notes on using S101 robot

## Find robot ports

python lerobot/scripts/find_motors_bus_port.py

## Find cameras

python -m lerobot.find_cameras opencv

## Calibrate

python -m lerobot.calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0  --robot.id s101_follower_v2
python -m lerobot.calibrate --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1 --teleop.id s101_leader_v2

## Teleoperate without cameras

python -m lerobot.teleoperate --robot.type so101_follower --robot.port /dev/ttyACM0  --robot.id s101_follower_v2 \
    --teleop.type so101_leader  --teleop.port /dev/ttyACM1 --teleop.id s101_leader_v2

python -m lerobot.teleoperate --config_path ./robot_config.yaml

## Teleoperate with cameras

python -m lerobot.teleoperate --robot.type so101_follower --robot.port /dev/ttyACM0  --robot.id s101_follower_v2 \
    --teleop.type so101_leader  --teleop.port /dev/ttyACM1 --teleop.id s101_leader_v2 \
    --robot.cameras "{ wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30}}" \
    --display_data true

python -m lerobot.teleoperate --config_path ./robot_config.yaml --display_data true

## Record


python -m lerobot.record \
    --config_path ./robot_config.yaml \
    --dataset.fps 30 \
    --dataset.single_task "Pick up the lego brick." \
    --dataset.repo_id fbeltrao/so101_multi_task \
    --dataset.tags '["so101","unplug cable", "lego"]' \
    --dataset.episode_time_s 60 \
    --dataset.reset_time_s 60 \
    --dataset.num_episodes 10 \
    --play_sounds true \
    --resume true \
    --dataset.push_to_hub false \
    --control.display_data true

## Visualize

python lerobot/scripts/visualize_dataset_html.py --repo-id fbeltrao/so101_multi_task

## Replay

python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=replay \
  --control.fps=30 \
  --control.repo_id=fbeltrao/so101_unplug_cable_4 \
  --control.episode=1



python -m lerobot.replay \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=s101_follower_v2 \
    --dataset.repo_id=fbeltrao/so101_multi_task \
    --dataset.episode=0

python -m lerobot.replay \
    --config_path ./robot_config.yaml \
    --dataset.repo_id fbeltrao/so101_multi_task \
    --dataset.episode 0 \
    --robot.cameras "{}"

## Hugging face login

huggingface-cli login

## Upload dataset from cli

huggingface-cli upload fbeltrao/so101_multi_task . --repo-type=dataset --revision steps_10_000

## Train on Azure ML compute instance

### First time installation

```plain
git clone https://github.com/fbeltrao/lerobot
cd lerobot
conda install ffmpeg=7.1.1 -c conda-forge
conda activate lerobot

git checkout francisco/s101-pi0
pip install -e ".[pi0]"
pip install transformers==4.48.1
pip install azureml-mlflow
```

### Once per login

Set training properties

```plain
export DATASET=so101_multi_task
export HF_USER=fbeltrao
export STEPS=5000
export POLICY_TYPE=pi0fast
```

```plain
git clone https://github.com/fbeltrao/lerobot
cd lerobot
conda activate lerobot
az login -t <tenant> --client-id "<principal-id>"
python lerobot/scripts/train.py --policy.type $POLICY_TYPE --dataset.repo_id "${HF_USER}/${DATASET}" --log_freq 100 --eval_freq 200 --steps $STEPS --output_dir ""/home/azureuser/cloudfiles/data/outputs/train/${DATASET}_${STEPS}steps" --job_name "${DATASET}_${STEPS}steps"
```

### Upload fine tuned model to Hugging face

huggingface-cli upload "${HF_USER}/${POLICY_TYPE}_${DATASET}" "/home/azureuser/cloudfiles/data/outputs/train/${DATASET}_${STEPS}steps/checkpoints/last/pretrained_model/" --revision "v${STEPS}steps"


## Understanding the inferencing

1. We first get observations from robot with `robot.get_observation()`

Return is a position of each joint, and images as np.array with RGB values.
```json
{
    'shoulder_pan.pos': -2.5246981339187613,
    'shoulder_lift.pos': -98.8125530110263,
    'elbow_flex.pos': 98.48282016956716,
    'wrist_flex.pos': 74.10444540353907,
    'wrist_roll.pos': -1.6308568470100937,
    'gripper.pos': 1.0107816711590296,
    'wrist': np.array(shape=(480, 640, 3), dtype=uint8),
    'side': np.array(shape=(480, 640, 3), dtype=uint8)
}
```

1. We get observation_features to generate the input for the policy. We get the values from  dataset (`dataset.features`) or robot configuration (`obs_features = hw_to_dataset_features(robot.observation_features, "observation")`). It maps `observation.state` to each joint, and each image to the corresponding `observation.image.x`

```python
obs_features = {
    'observation.state': {'dtype': 'float32', 'shape': (6,), 'names': ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']}, 
    'observation.images.wrist': {'dtype': 'video', 'shape': (480, 640, 3), 'names': ['height', 'width', 'channels']}, 
    'observation.images.side': {'dtype': 'video', 'shape': (480, 640, 3), 'names': ['height', 'width', 'channels']}}
```

1. We generate the initial input for the policy using `build_dataset_frame(obs_features, obs, prefix="observation")`
```python

dataset_frame = build_dataset_frame(obs_features, obs, prefix="observation")

# dataset_frame:
{   
    'observation.state': np.array([ -2.524698, -98.81255, 98.48282, 74.10445,  -1.6308569, 1.0107816], dtype=float32), 
    'observation.images.wrist': np.array(shape=(480, 640, 3), dtype=uint8),
    'observation.images.side': np.array(shape=(480, 640, 3), dtype=uint8),
}

```

1. We call `predict_action(observation: dict[str, np.ndarray], policy: PreTrainedPolicy, device: torch.device, use_amp: bool, task: str | None=None, robot_type: str | None = None)`. Internally it will convert input to tensors and add robot_type and task.

```python
# Will convert all fields to torch (`from_numpy` and `unsqueeze(0)`). Images will be `/255` and `permute(2,0,1).contiguous()`.

observation = {
    'task': 'Pick up the lego brick.',
    'robot_type': 'so101_follower',
    'observation.state': torch.Tensor(shape=[1, 6]),
    'observation.images.wrist': torch.Tensor(shape=[1, 3, 480, 640]),
    'observation.images.side': torch.Tensor(shape=[1, 3, 480, 640]),
}

action = policy.select_action(observation) # shape = [1,6]
action = action.squeeze(0) # shape = [6]
```

1. Transform actions tensor (shape=[6]) to robot specific actions (joint.pos) then call `robot.send_action`
```python
action_values = predict_action(...)
action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}

# action = { # all values are float
#   'shoulder_pan.pos': -5.473992839917791, 
#   'shoulder_lift.pos': -17.40812786691692, 
#   'elbow_flex.pos': 35.52430382950564, 
#   'wrist_flex.pos': 60.002333510349644, 
#   'wrist_roll.pos': 1.6304100696448507, 
#   'gripper.pos': 2.3112573960823086
# }