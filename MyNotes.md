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

## Episodes to remove

## Upload fine tuned model to Hugging face

huggingface-cli upload fbeltrao/pi0fast_so101_unplug_cable outputs/train/s101_unplug_cable_4_1000steps/checkpoints/last/pretrained_model/ --revision v1000steps

## Hugging face login

huggingface-cli login

## Upload dataset from cli

huggingface-cli upload fbeltrao/so101_unplug_cable_4 . --repo-type=dataset --revision steps_10_000

## Train

git clone https://github.com/fbeltrao/lerobot
cd lerobot && \
  conda activate lerobot && \
  git checkout francisco/s101-pi0
pip install azureml-mlflow

az login -t <tenant> --client-id "<principal-id>"

python lerobot/scripts/train.py --policy.type pi0fast --dataset.repo_id fbeltrao/so101_unplug_cable_4 --log_freq 100 --eval_freq 200 --steps 10000 --output_dir outputs/train/so101_unplug_cable_4_10000steps --job_name so101_unplug_cable_4_10000steps