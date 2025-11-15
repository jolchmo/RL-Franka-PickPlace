#!/usr/bin/env python3
# train_isaaclab.py
# 使用Isaac Lab + Stable-Baselines3 训练Franka抓取任务



import argparse
import os
from datetime import datetime

# Isaac Lab导入
try:
    from isaaclab.app import AppLauncher
except ImportError:
    from omni.isaac.lab.app import AppLauncher

# 解析命令行参数
parser = argparse.ArgumentParser(description="训练Franka抓取任务")
parser.add_argument("--num_envs", type=int, default=2048, help="并行环境数量")
parser.add_argument("--task", type=str, default="FrankaPickPlace", help="任务名称")
parser.add_argument("--headless", action="store_true", help="无头模式")
parser.add_argument("--test", action="store_true", help="测试模式")
parser.add_argument("--checkpoint", type=str, default=None, help="检查点路径")
parser.add_argument("--timesteps", type=int, default=10000000, help="训练总步数")
args_cli = parser.parse_args()

# 启动Isaac Lab应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 在启动后导入其他模块
import torch
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

try:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
    from isaaclab.envs.mdp.actions import JointPositionActionCfg
except ImportError:
    from omni.isaac.lab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
    from omni.isaac.lab.envs.mdp.actions import JointPositionActionCfg

from franka_pickplace_env import FrankaPickPlaceEnv
from franka_pickplace_env_cfg import FrankaPickPlaceEnvCfg


# =================================================================
# Isaac Lab环境包装器 for Stable-Baselines3
# =================================================================
class IsaacLabWrapper(gym.Wrapper):
    """将Isaac Lab环境包装为标准Gymnasium接口"""
    
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space["policy"]
        self.action_space = env.action_space
        
    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        return obs_dict["policy"], info
        
    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        return obs_dict["policy"], reward, terminated, truncated, info


# =================================================================
# 训练函数（使用Stable-Baselines3）
# =================================================================
def train_with_sb3():
    """生成rl-games训练配置"""
    return {
        "params": {
            "seed": 42,
            "algo": {"name": "a2c_continuous"},
            "model": {"name": "continuous_a2c_logstd"},
            "network": {
                # 使用最简单的共享网络配置
                "name": "actor_critic",
                "separate": False, # 恢复为 False，使用共享网络
                "space": {
                    "continuous": {
                        "mu_activation": "None", "sigma_activation": "None",
                        "mu_init": {"name": "default"},
                        "sigma_init": {"name": "const_initializer", "val": 0.0},
                        "fixed_sigma": True,
                    }
                },
                "mlp": {
                    "units": [256, 128, 64], "activation": "elu", "d2rl": False,
                    "initializer": {"name": "default"},
                },
            },
            "config": {
                "name": "FrankaPickPlace", "env_name": "rlgpu", "ppo": True,
                "mixed_precision": False, 
                "normalize_input": True,
                # vvvvvv 这是解决所有问题的核心 vvvvvv
                "normalize_value": False, # 直接禁用价值函数归一化，避免所有问题
                # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                "value_bootstrap": True, "num_actors": num_envs,
                "reward_shaper": {"scale_value": 0.01}, "normalize_advantage": True,
                "gamma": 0.99, "tau": 0.95, "learning_rate": 3e-4,
                "lr_schedule": "adaptive", "kl_threshold": 0.016,
                "max_epochs": max_iterations, "save_best_after": 100,
                "save_frequency": 50, "print_stats": True, "grad_norm": 1.0,
                "entropy_coef": 0.0, "e_clip": 0.2, "horizon_length": 16,
                "minibatch_size": 8192, "mini_epochs": 8, "critic_coef": 4,
                "clip_value": True, "train_dir": checkpoint_dir,
            },
        }
    }

# =================================================================
# <<< 釜底抽薪方案 PART 2: 恢复最简单的环境包装器 >>>
# =================================================================
class IsaacLabVecEnvWrapper(vecenv.IVecEnv):
    """
    一个简单的包装器，将Isaac Lab环境适配为rl-games格式。
    它只返回一个扁平的观察张量。
    """
    
    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs
        obs_space = env.observation_space["policy"]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=obs_space.shape[1:],  # 使用 obs_space 的形状
            dtype=np.float32
        )
        self.action_space = env.action_space
        self.num_agents = self.num_envs
        
    def step(self, actions):
        obs_dict, rewards, dones, truncated, info = self.env.step(actions)
        # 只返回策略需要的张量
        return obs_dict["policy"], rewards, dones, info
        
    def reset(self, **kwargs):
        obs_dict, _ = self.env.reset(**kwargs)
        # 只返回策略需要的张量
        return obs_dict["policy"]
        
    def get_number_of_agents(self):
        return self.num_envs
        
    def get_env_info(self):
        # 返回简单的 Box 空间信息
        return {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "agents": self.num_envs,
        }

def main():
    print("=" * 60)
    print("🤖 Isaac Lab - Franka抓取任务训练")
    print(f"环境数量: {args_cli.num_envs}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)
    
    
    simulation_app.close()

if __name__ == "__main__":
    main()

