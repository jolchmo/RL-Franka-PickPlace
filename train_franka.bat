@echo off
REM Franka PickPlace 训练脚本 - 改进版 v2


REM 使用 isaaclab.bat 启动器（自动设置 ISAACLAB_ASSETS_DIR 等环境变量）
C:\IsaacSim\python.bat C:\Allfile\WS\RL-Franka-PickPlace\scripts\rsl_rl\train.py --task FrankaPickPlace --num_envs 32 --max_iterations 10000 --headless


pause
