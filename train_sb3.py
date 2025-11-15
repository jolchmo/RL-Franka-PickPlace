#!/usr/bin/env python3
# train_sb3.py
# 使用Isaac Lab + Stable-Baselines3 训练Franka抓取任务

"""
推荐配置:
- 低显存 (8GB):   --num_envs 2048
- 中等显存 (16GB): --num_envs 4096
- 高显存 (24GB):   --num_envs 8192
- 超高显存 (48GB): --num_envs 16384

使用方法:
# 训练（2048个并行环境）
python train_sb3.py --num_envs 2048 --headless --timesteps 10000000

# 测试训练好的模型
python train_sb3.py --num_envs 16 --test --checkpoint models/franka_pickplace.zip
"""

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
parser.add_argument("--total_timesteps", type=int, default=10000000, help="训练总步数")
args_cli = parser.parse_args()

# 启动Isaac Lab应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 在启动后导入其他模块
import torch
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure

from franka_pickplace_env import FrankaPickPlaceEnv
from franka_pickplace_env_cfg import FrankaPickPlaceEnvCfg


# =================================================================
# 自定义回调：记录详细训练信息
# =================================================================
class DetailedLoggingCallback(BaseCallback):
    """记录每个rollout的平均奖励和其他统计信息"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # 获取当前的奖励
        if "reward" in self.locals:
            rewards = self.locals["rewards"]
            # 记录平均奖励
            if len(rewards) > 0:
                mean_reward = np.mean(rewards)
                self.logger.record("rollout/mean_reward_step", mean_reward)
        
        return True
    
    def _on_rollout_end(self) -> bool:
        """在每个rollout结束时记录统计信息"""
        # 记录rollout buffer中的信息
        if hasattr(self.model, 'rollout_buffer'):
            buffer = self.model.rollout_buffer
            if buffer.full:
                mean_reward = np.mean(buffer.rewards)
                self.logger.record("rollout/mean_reward", mean_reward)
                self.logger.record("rollout/mean_value", np.mean(buffer.values))
        
        return True


# =================================================================
# Isaac Lab环境包装器 for Stable-Baselines3
# =================================================================
from stable_baselines3.common.vec_env import VecEnv


class IsaacLabVecEnvWrapper(VecEnv):
    """将Isaac Lab向量化环境包装为SB3的VecEnv接口"""
    
    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs
        
        # 使用 policy 观察空间，移除批次维度
        original_obs_space = env.observation_space["policy"]
        if len(original_obs_space.shape) == 2:
            # 形状是 (num_envs, obs_dim)，取单个环境的维度
            obs_shape = (original_obs_space.shape[1],)
        else:
            # 形状已经是 (obs_dim,)
            obs_shape = original_obs_space.shape
        
        observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )
        
        # 修复动作空间边界（SB3 需要有限边界），移除批次维度
        original_action_space = env.action_space
        if len(original_action_space.shape) == 2:
            # 形状是 (num_envs, action_dim)，取单个环境的维度
            action_shape = (original_action_space.shape[1],)
        else:
            # 形状已经是 (action_dim,)
            action_shape = original_action_space.shape
        
        action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=action_shape,
            dtype=np.float32
        )
        
        # 初始化VecEnv
        super().__init__(self.num_envs, observation_space, action_space)
        
        self.reset_infos = [{} for _ in range(self.num_envs)]
    
    def _to_numpy(self, tensor):
        """将torch tensor转换为numpy数组（自动处理GPU/CPU）"""
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()
        return np.array(tensor)
    
    def reset(self):
        obs_dict, info = self.env.reset()
        obs = self._to_numpy(obs_dict["policy"])
        return obs
    
    def step_async(self, actions):
        """异步执行动作（Isaac Lab是同步的，所以直接存储）"""
        self.actions = actions
    
    def step_wait(self):
        """等待step完成并返回结果"""
        # 转换action为torch tensor，确保是2D
        if isinstance(self.actions, np.ndarray):
            actions = torch.from_numpy(self.actions).float()
        else:
            actions = torch.as_tensor(self.actions, dtype=torch.float32)
        
        # 确保形状正确：应该是 (num_envs, action_dim)
        if actions.shape[0] != self.num_envs:
            raise ValueError(f"Action batch size mismatch: expected {self.num_envs}, got {actions.shape[0]}")
        
        actions = actions.to(self.env.device)
        
        obs_dict, rewards, terminated, truncated, infos = self.env.step(actions)
        
        # 转换所有输出为numpy数组
        obs = self._to_numpy(obs_dict["policy"])
        rewards = self._to_numpy(rewards)
        
        # 先转换为numpy再进行逻辑运算
        terminated_np = self._to_numpy(terminated)
        truncated_np = self._to_numpy(truncated)
        dones = np.logical_or(terminated_np, truncated_np)
        
        # SB3期望infos是一个字典列表
        if not isinstance(infos, list):
            infos = [{} for _ in range(self.num_envs)]
        
        return obs, rewards, dones, infos
    
    def close(self):
        """关闭环境"""
        self.env.close()
    
    def env_is_wrapped(self, wrapper_class, indices=None):
        """检查环境是否被包装"""
        return [False] * self.num_envs
    
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """调用环境方法"""
        return [getattr(self.env, method_name)(*method_args, **method_kwargs)]
    
    def get_attr(self, attr_name, indices=None):
        """获取环境属性"""
        return [getattr(self.env, attr_name)]
    
    def set_attr(self, attr_name, value, indices=None):
        """设置环境属性"""
        setattr(self.env, attr_name, value)
    
    def seed(self, seed=None):
        """设置随机种子"""
        if seed is not None:
            self.env.seed(seed)
        return [seed] * self.num_envs


# =================================================================
# 训练函数
# =================================================================
def train():
    """使用Stable-Baselines3训练"""
    print("=" * 60)
    print("🚀 使用Stable-Baselines3训练（GPU加速）")
    print("=" * 60)
    
    # 创建环境配置
    env_cfg = FrankaPickPlaceEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # 创建环境
    print(f"📦 创建 {args_cli.num_envs} 个并行环境...")
    print(f"🔧 配置确认: env_cfg.scene.num_envs = {env_cfg.scene.num_envs}")
    
    # 检查GPU显存
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        total_mem_gb = gpu_props.total_memory / 1024**3
        print(f"🎮 GPU: {gpu_props.name}")
        print(f"💾 总显存: {total_mem_gb:.2f} GB")
        allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
        print(f"📊 已分配: {allocated_gb:.2f} GB")
    
    base_env = FrankaPickPlaceEnv(cfg=env_cfg)
    
    # 检查环境创建后的显存
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
        reserved_gb = torch.cuda.memory_reserved(0) / 1024**3
        print(f"💾 环境创建后显存: 分配={allocated_gb:.2f}GB, 保留={reserved_gb:.2f}GB")
    
    print(f"✅ 实际创建的环境数: {base_env.num_envs}")
    
    # 检查环境数是否匹配
    if base_env.num_envs != args_cli.num_envs:
        print("\n" + "="*70)
        print("⚠️  警告: 环境数不匹配!")
        print(f"   请求环境数: {args_cli.num_envs}")
        print(f"   实际环境数: {base_env.num_envs}")
        print(f"   差异: {args_cli.num_envs - base_env.num_envs} 个环境未创建")
        print("\n   可能原因:")
        print("   1. Isaac Lab 内部有最大环境数限制")
        print("   2. 配置文件的默认值覆盖了命令行参数")
        print("   3. 显存不足，系统自动降级")
        print("="*70 + "\n")
        
        user_input = input("是否继续训练? (y/n): ")
        if user_input.lower() != 'y':
            print("训练已取消")
            return
    else:
        print(f"✅ 环境数匹配! 成功创建了所有 {base_env.num_envs} 个环境\n")
    
    env = IsaacLabVecEnvWrapper(base_env)
    
    print(f"🔍 观察空间: {env.observation_space}")
    print(f"🔍 动作空间: {env.action_space}")
    print(f"🔍 最终并行环境数: {env.num_envs}")
    
    # 准备输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/sb3_{timestamp}"
    model_dir = f"./models"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 配置回调
    checkpoint_callback = CheckpointCallback(
        save_freq=1000000 // args_cli.num_envs,  # 每1000k步保存一次
        save_path=model_dir,
        name_prefix="franka_pickplace",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    logging_callback = DetailedLoggingCallback(verbose=1)
    
    from stable_baselines3.common.callbacks import CallbackList
    callbacks = CallbackList([checkpoint_callback, logging_callback])
    
    # 创建或加载模型
    if args_cli.checkpoint and os.path.exists(args_cli.checkpoint):
        print(f"📂 从检查点加载: {args_cli.checkpoint}")
        model = PPO.load(
            args_cli.checkpoint,
            env=env,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        print("🆕 创建新模型")
        # 根据环境数动态调整batch_size
        num_envs = args_cli.num_envs
        # batch_size应该是num_envs的倍数，且足够大以充分利用GPU
        if num_envs >= 8192:
            batch_size = 8192
            n_steps = 16
        elif num_envs >= 4096:
            batch_size = 4096
            n_steps = 32
        else:
            batch_size = 2048
            n_steps = 32
        
        print(f"📊 训练配置: num_envs={num_envs}, n_steps={n_steps}, batch_size={batch_size}")
        print(f"   每次更新收集: {num_envs * n_steps} 步")
        print(f"   每次更新梯度步数: {(num_envs * n_steps) // batch_size * 10} 步")
        
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=10,  # 每次更新训练10个epoch
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # 鼓励探索
            vf_coef=0.5,  # 价值函数损失权重
            max_grad_norm=0.5,  # 梯度裁剪
            verbose=1,
            tensorboard_log=log_dir,
            device="cuda" if torch.cuda.is_available() else "cpu",
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),
                activation_fn=torch.nn.ELU,
            ),
        )
    
    # 配置日志
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    
    print(f"🎯 开始训练 ({args_cli.total_timesteps} 总步数)")
    print(f"💾 模型保存到: {model_dir}")
    print(f"📊 日志保存到: {log_dir}")
    print(f"📈 TensorBoard: tensorboard --logdir={log_dir}")
    print("=" * 60)
    
    # 显示训练开始前的显存
    if torch.cuda.is_available():
        print(f"\n💾 训练前显存: {torch.cuda.memory_reserved(0) / 1024**3:.2f}GB")
        print("⏳ 训练开始后显存会增长，请用 nvidia-smi 监控\n")
    
    # 开始训练
    model.learn(
        total_timesteps=args_cli.total_timesteps,
        callback=callbacks,
        progress_bar=True,
        log_interval=1,  # 每个rollout后记录日志
    )
    
    # 保存最终模型
    final_model_path = f"{model_dir}/franka_pickplace_final.zip"
    model.save(final_model_path)
    print(f"✅ 训练完成! 最终模型保存到: {final_model_path}")
    
    env.close()


# =================================================================
# 测试函数
# =================================================================
def test():
    """测试训练好的模型"""
    if not args_cli.checkpoint:
        print("❌ 测试模式需要指定 --checkpoint 参数")
        return
        
    if not os.path.exists(args_cli.checkpoint):
        print(f"❌ 检查点不存在: {args_cli.checkpoint}")
        return
    
    print("=" * 60)
    print("🧪 测试模式")
    print("=" * 60)
    
    # 创建环境
    env_cfg = FrankaPickPlaceEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    base_env = FrankaPickPlaceEnv(cfg=env_cfg)
    env = IsaacLabVecEnvWrapper(base_env)
    
    # 加载模型
    print(f"📂 加载模型: {args_cli.checkpoint}")
    model = PPO.load(args_cli.checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # 运行测试
    print("▶️ 开始测试...")
    obs = env.reset()  # VecEnv.reset() 只返回观察
    episode_rewards = []
    current_rewards = np.zeros(env.num_envs)
    
    for step in range(1000):  # 测试1000步
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)
        current_rewards += reward
        
        # 记录完成的episode
        if dones.any():
            episode_rewards.extend(current_rewards[dones].tolist())
            current_rewards[dones] = 0
            
        if step % 100 == 0:
            print(f"步数: {step}, 平均奖励: {np.mean(current_rewards):.2f}")
    
    if episode_rewards:
        print(f"\n✅ 测试完成!")
        print(f"完成的episode数: {len(episode_rewards)}")
        print(f"平均奖励: {np.mean(episode_rewards):.2f}")
        print(f"最大奖励: {np.max(episode_rewards):.2f}")
        print(f"最小奖励: {np.min(episode_rewards):.2f}")
    
    env.close()


# =================================================================
# 主函数
# =================================================================
def main():
    print("=" * 60)
    print("🤖 Isaac Lab + Stable-Baselines3 - Franka抓取任务")
    print("=" * 60)
    print(f"环境数量: {args_cli.num_envs}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)
    
    if args_cli.test:
        test()
    else:
        train()
    
    simulation_app.close()


if __name__ == "__main__":
    main()
