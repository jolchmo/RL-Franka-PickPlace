#!/bin/bash
# Franka PickPlace 视频录制脚本 - Linux 服务器版

# 服务器上 Isaac Sim 的路径
ISAACSIM_PATH="/opt/IsaacSim"

# 项目路径
PROJECT_PATH="/home/user/RL-Franka-PickPlace"

# 日志目录
LOG_DIR="$PROJECT_PATH/logs/rsl_rl/franka_pick_place"

# 查找最新训练目录
LATEST_DIR=$(ls -td "$LOG_DIR"/*/ 2>/dev/null | head -1 | xargs basename)

if [ -z "$LATEST_DIR" ]; then
    echo "错误: 找不到训练目录"
    exit 1
fi

echo "找到最新训练目录: $LATEST_DIR"

CHECKPOINT_DIR="$LOG_DIR/$LATEST_DIR"

# 查找最新模型
LATEST_MODEL=$(ls -t "$CHECKPOINT_DIR"/model_*.pt 2>/dev/null | head -1)

if [ -z "$LATEST_MODEL" ]; then
    echo "错误: 找不到模型文件"
    exit 1
fi

echo "使用模型: $(basename $LATEST_MODEL)"

# 录制视频
$ISAACSIM_PATH/python.sh $PROJECT_PATH/scripts/rsl_rl/play.py \
    --task FrankaPickPlace \
    --num_envs 16 \
    --load_run "$LATEST_DIR" \
    --video \
    --video_length 1000 \
    --headless

echo ""
echo "============================================"
echo "视频录制完成！"
echo "查看: $LOG_DIR/$LATEST_DIR/videos/play/"
echo "============================================"
