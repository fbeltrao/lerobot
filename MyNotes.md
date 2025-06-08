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

CKPT=last
huggingface-cli upload ${HF_USER}/act_so101_test${CKPT} \
  outputs/train/act_so101_test/checkpoints/${CKPT}/pretrained_model


# Hugging face login

huggingface-cli login


# Upload dataset from cli

huggingface-cli upload fbeltrao/so101_unplug_cable_4 . --repo-type=dataset
