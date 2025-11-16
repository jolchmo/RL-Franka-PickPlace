#!/usr/bin/env python3
"""
奖励计算验证脚本
目的：验证以下关键问题：
1. 奖励函数是否真的被 RewardManager 调用？
2. 奖励值是否在变化，还是一直为0？
3. 奖励是否正确传递给了 SB3 的 PPO 算法？
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="验证奖励计算")
parser.add_argument("--num_envs", type=int, default=128, help="并行环境数")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from franka_pickplace_env import FrankaPickPlaceEnv
from franka_pickplace_env_cfg import FrankaPickPlaceEnvCfg

def main():
    print("=" * 70)
    print("🔍 奖励计算验证测试")
    print("=" * 70)
    print("\n目标：验证奖励函数是否被调用，以及奖励值是否变化\n")
    
    # 创建环境
    env_cfg = FrankaPickPlaceEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    print(f"📦 创建 {args_cli.num_envs} 个环境...")
    env = FrankaPickPlaceEnv(cfg=env_cfg)
    
    print(f"✅ 环境创建成功")
    print(f"   观测维度: {env.observation_manager.group_obs_dim['policy']}")
    print(f"   动作维度: {env.action_manager.total_action_dim}")
    print(f"\n🎯 开始测试 - 运行1000步，每100步统计一次奖励\n")
    print("-" * 70)
    
    # 重置环境
    obs, _ = env.reset()
    
    total_rewards = []
    step_rewards = []
    
    for step in range(1000):
        # 生成随机动作
        actions = torch.randn(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        actions = torch.clamp(actions, -1.0, 1.0)
        
        # 执行步骤
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # 收集奖励统计
        step_rewards.append(rewards.mean().item())
        total_rewards.extend(rewards.cpu().numpy().tolist())
        
        # 每100步打印一次统计
        if (step + 1) % 100 == 0:
            recent_mean = sum(step_rewards[-100:]) / 100
            recent_std = torch.tensor(step_rewards[-100:]).std().item()
            print(f"步数 {step+1:4d}: 最近100步平均奖励={recent_mean:+.6f} ± {recent_std:.6f}")
    
    print("\n" + "=" * 70)
    print("📊 最终统计")
    print("=" * 70)
    
    import numpy as np
    total_rewards = np.array(total_rewards)
    
    print(f"\n总步数: {len(step_rewards)}")
    print(f"平均奖励: {np.mean(total_rewards):+.6f}")
    print(f"标准差: {np.std(total_rewards):.6f}")
    print(f"最小奖励: {np.min(total_rewards):+.6f}")
    print(f"最大奖励: {np.max(total_rewards):+.6f}")
    print(f"奖励为0的比例: {(total_rewards == 0).sum() / len(total_rewards) * 100:.1f}%")
    
    # 关键诊断
    print("\n" + "=" * 70)
    print("🔍 诊断结果")
    print("=" * 70)
    
    if not hasattr(env, '_reward_call_count'):
        print("\n❌ 问题1: 奖励函数从未被调用!")
        print("   可能原因: RewardManager配置错误，或环境没有正确初始化")
    else:
        print(f"\n✅ 奖励函数被调用了 {env._reward_call_count} 次")
    
    if np.all(total_rewards == total_rewards[0]):
        print("\n❌ 问题2: 所有奖励值完全相同 (constant)!")
        print(f"   固定值: {total_rewards[0]:.6f}")
        print("   可能原因: 奖励函数内部逻辑有问题，返回了常量")
    elif np.std(total_rewards) < 1e-6:
        print("\n⚠️  问题2: 奖励值几乎不变化 (std < 1e-6)!")
        print("   可能原因: 奖励函数对状态变化不敏感")
    else:
        print(f"\n✅ 奖励值在变化 (std={np.std(total_rewards):.6f})")
    
    if (total_rewards == 0).sum() / len(total_rewards) > 0.95:
        print("\n❌ 问题3: 超过95%的奖励为0!")
        print("   可能原因: 奖励条件太苛刻，智能体几乎无法获得正奖励")
    else:
        print(f"\n✅ 奖励分布正常 ({(total_rewards != 0).sum() / len(total_rewards) * 100:.1f}%非零)")
    
    print("\n" + "=" * 70)
    print("✅ 验证测试完成")
    print("=" * 70)
    
    # 查看调试输出的提示
    print("\n💡 提示: 向上滚动查看 '[REWARD FUNC CALLED]' 开头的调试输出")
    print("   如果看到这些输出，说明奖励函数确实在运行")
    print("   如果没有，说明 RewardManager 根本没有调用我们的函数\n")
    
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
