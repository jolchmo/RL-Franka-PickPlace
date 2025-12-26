#!/bin/bash
# Franka PickPlace 训练脚本 - Linux 服务器版

# 设置 Isaac Lab 环境变量（根据服务器实际路径修改）
# export ISAACLAB_ASSETS_DIR="/path/to/isaaclab_assets"
# export ISAACLAB_DATASETS_DIR="/path/to/isaaclab_datasets"

# 服务器上 Isaac Sim 的路径
ISAACSIM_PATH="/opt/IsaacSim"

# 项目路径
PROJECT_PATH="/home/user/RL-Franka-PickPlace"

# 使用 isaaclab.sh 启动器
$ISAACSIM_PATH/python.sh $PROJECT_PATH/scripts/rsl_rl/train.py \
    --task FrankaPickPlace \
    --num_envs 64 \
    --max_iterations 10000 \
    --headless \
    --device cuda
