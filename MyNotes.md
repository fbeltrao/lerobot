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
