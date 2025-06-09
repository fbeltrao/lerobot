# Find robot ports

python lerobot/scripts/find_motors_bus_port.py

# Find cameras

python lerobot/common/robot_devices/cameras/opencv.py \
    --images-dir outputs/images_from_opencv_cameras

# Teleoperate with cameras

python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=teleoperate \
  --control.display_data=true


# Record

python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Unplug the cable." \
  --control.repo_id=fbeltrao/so101_unplug_cable_3 \
  --control.tags='["so101","unplug cable"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=20 \
  --control.reset_time_s=30 \
  --control.num_episodes=10 \
  --control.resume=true \
  --control.display_data=true


# Visualize

python lerobot/scripts/visualize_dataset_html.py --repo-id fbeltrao/so101_unplug_cable_3

# Replay

python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=replay \
  --control.fps=30 \
  --control.repo_id=fbeltrao/so101_unplug_cable_3 \
  --control.episode=1



 # Episodes to remove


# Upload fine tuned model to Hugging face

huggingface-cli upload fbeltrao/pi0fast_so101_unplug_cable outputs/train/s101_unplug_cable_4_1000steps/checkpoints/last/pretrained_model/ --revision v1000steps


# Hugging face login

huggingface-cli login


# Upload dataset from cli

huggingface-cli upload fbeltrao/so101_unplug_cable_4 . --repo-type=dataset


# Train

git clone https://github.com/fbeltrao/lerobot
cd lerobot
conda activate lerobot
git checkout francisco/s101-pi0
pip install azureml-mlflow
pip install mlflow

python lerobot/scripts/train.py --policy.type pi0fast --dataset.repo_id fbeltrao/so101_unplug_cable_4 --log_freq 50 --eval_freq 100 --steps 5000 --output_dir outputs/train/so101_unplug_cable_4_5000steps --job_name so101_unplug_cable_4_5000steps
