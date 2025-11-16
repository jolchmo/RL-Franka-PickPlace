#!/usr/bin/env python3
"""
夹爪控制验证脚本
用于可视化测试夹爪是否真的在响应动作指令
"""

import argparse
from isaaclab.app import AppLauncher

# 解析命令行参数
parser = argparse.ArgumentParser(description="测试Franka夹爪控制")
parser.add_argument("--num_envs", type=int, default=4, help="并行环境数量")
args_cli = parser.parse_args()

# 启动Isaac Lab应用（非headless模式，需要可视化）
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from franka_pickplace_env import FrankaPickPlaceEnv
from franka_pickplace_env_cfg import FrankaPickPlaceEnvCfg

def main():
    """主测试函数"""
    print("=" * 60)
    print("🔧 Franka夹爪控制验证测试")
    print("=" * 60)
    
    # 创建环境配置
    env_cfg = FrankaPickPlaceEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # 创建环境
    print(f"\n📦 创建 {args_cli.num_envs} 个可视化环境...")
    env = FrankaPickPlaceEnv(cfg=env_cfg)
    
    print("\n✅ 环境创建成功！")
    print(f"   观测空间维度: {env.observation_manager.group_obs_dim['policy']}")
    print(f"   动作空间维度: {env.action_manager.total_action_dim}")
    
    # 重置环境
    obs, _ = env.reset()
    
    print("\n🎮 开始测试序列...")
    print("   测试1: 所有动作为0 (夹爪应保持打开)")
    print("   测试2: 夹爪动作=+1 (夹爪应尝试闭合)")
    print("   测试3: 夹爪动作=-1 (夹爪应打开)")
    print("\n请仔细观察仿真窗口中的机器人夹爪！\n")
    
    # 测试序列
    test_steps = 100
    
    # 阶段1：所有动作为0（50步）
    print("🔹 阶段1: 零动作测试...")
    for i in range(50):
        actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        obs, reward, terminated, truncated, info = env.step(actions)
        if i % 10 == 0:
            print(f"  Step {i}: 夹爪动作值 = 0.0")
    
    # 阶段2：夹爪闭合指令（50步）
    print("\n🔹 阶段2: 夹爪闭合测试 (action[7] = +1.0)...")
    for i in range(50):
        actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        # 设置第8个动作（索引7）为正值，指示闭合
        if env.action_manager.total_action_dim >= 8:
            actions[:, 7] = 1.0
        elif env.action_manager.total_action_dim >= 9:
            actions[:, 7] = 1.0  # 或者 actions[:, 8] = 1.0，取决于配置
        obs, reward, terminated, truncated, info = env.step(actions)
        if i % 10 == 0:
            print(f"  Step {i+50}: 夹爪动作值 = +1.0 (闭合指令)")
    
    # 阶段3：夹爪打开指令（50步）
    print("\n🔹 阶段3: 夹爪打开测试 (action[7] = -1.0)...")
    for i in range(50):
        actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        if env.action_manager.total_action_dim >= 8:
            actions[:, 7] = -1.0
        elif env.action_manager.total_action_dim >= 9:
            actions[:, 7] = -1.0
        obs, reward, terminated, truncated, info = env.step(actions)
        if i % 10 == 0:
            print(f"  Step {i+100}: 夹爪动作值 = -1.0 (打开指令)")
    
    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("\n请回答以下问题：")
    print("1. 在阶段2中，您是否看到夹爪的两个指片有向内闭合的动作？")
    print("2. 在阶段3中，夹爪是否重新打开？")
    print("\n如果答案是'是'，说明夹爪控制正常。")
    print("如果答案是'否'，说明动作空间配置有问题，需要检查ActionsCfg。")
    print("=" * 60)
    
    # 关闭环境
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
