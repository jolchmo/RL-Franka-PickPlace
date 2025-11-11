"""
RL一体化脚本 - 训练、测试、运行全流程
包含三种模式：
1. 训练模式 (train): 训练RL模型
2. 测试模式 (test): 测试训练好的模型
3. 运行模式 (run): 使用RL模型运行任务
"""

import os
import argparse
import numpy as np

# ============================================================================
# 工具函数
# ============================================================================

def check_dependencies():
    """检查必要的依赖包"""
    required_packages = {
        'gymnasium': 'gymnasium>=0.29.0',
        'stable_baselines3': 'stable-baselines3>=2.0.0',
        'torch': 'torch>=2.0.0',
    }
    
    missing = []
    for pkg_name, pkg_version in required_packages.items():
        try:
            __import__(pkg_name)
        except ImportError:
            missing.append(pkg_version)
    
    if missing:
        print("=" * 60)
        print("❌ 缺少必要的依赖包:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\n请运行以下命令安装:")
        print("   pip install -r requirements_rl.txt")
        print("=" * 60)
        return False
    return True


# ============================================================================
# 模式1: 训练模式
# ============================================================================

def train_model(args):
    """训练RL模型"""
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    
    print("=" * 60)
    print(f"🚀 开始训练 - 算法: {args.algorithm.upper()}")
    print("=" * 60)
    print(f"训练步数: {args.timesteps}")
    print(f"方块数量: {args.cube_num}")
    print(f"无头模式: {args.headless}")
    print("=" * 60)
    
    # ✅ 先初始化 Isaac Sim
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": args.headless})
    
    # ✅ 再导入依赖 Isaac Sim 的模块
    from class_taskEnv import ArmPickPlaceRLEnv
    
    # 创建环境
    def make_env():
        env = ArmPickPlaceRLEnv(
            headless=args.headless, 
            cube_num=args.cube_num,
            simulation_app=simulation_app
        )
        return Monitor(env)
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env)
    
    # 创建保存目录
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    # 设置checkpoint回调
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/",
        name_prefix=f"{args.algorithm}_armpickplace"
    )
    
    # 创建模型（优化训练速度）
    if args.algorithm == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,  
            batch_size=128,  
            n_epochs=10,  
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            verbose=1,
            tensorboard_log="./logs/",
            policy_kwargs=dict(
                net_arch=[dict(pi=[256, 256], vf=[128, 128])]  #
            )
        )
    else:
        raise ValueError(f"不支持的算法: {args.algorithm}")
    
    print("\n开始训练...")
    print("提示: 使用 'tensorboard --logdir ./logs/' 查看训练进度")
    print("-" * 60)
    
    # 训练
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # 保存最终模型
    final_model_path = f"./models/{args.algorithm}_armpickplace_final"
    model.save(final_model_path)
    env.save(f"./models/vec_normalize_{args.algorithm}.pkl")
    
    print("\n" + "=" * 60)
    print("✅ 训练完成!")
    print(f"模型已保存到: {final_model_path}.zip")
    print(f"VecNormalize已保存到: ./models/vec_normalize_{args.algorithm}.pkl")
    print("\n下一步:")
    print(f"  测试模型: python {__file__} test --model {final_model_path}.zip")
    print(f"  运行模型: python {__file__} run --model {final_model_path}.zip")
    print("=" * 60)
    
    env.close()


# ============================================================================
# 模式2: 测试模式
# ============================================================================

def test_model(args):
    """测试训练好的模型"""
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    
    print("=" * 60)
    print(f"🧪 测试模型 - {args.model}")
    print("=" * 60)
    
    # ✅ 先初始化 Isaac Sim
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": args.headless})
    
    # ✅ 再导入依赖 Isaac Sim 的模块
    from class_taskEnv import ArmPickPlaceRLEnv
    
    # 创建环境
    env = ArmPickPlaceRLEnv(
        render_mode="human",
        headless=args.headless, 
        cube_num=args.cube_num,
        simulation_app=simulation_app
    )
    
    # 加载VecNormalize（如果提供）
    if args.vec_normalize:
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load(args.vec_normalize, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        vec_env = None
    
    # 加载模型
    if args.algorithm == "ppo":
        model = PPO.load(args.model)
    elif args.algorithm == "sac":
        model = SAC.load(args.model)
    else:
        raise ValueError(f"不支持的算法: {args.algorithm}")
    
    print(f"测试 {args.episodes} 个回合...")
    print("-" * 60)
    
    # 统计信息
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(args.episodes):
        obs, info = env.reset()
        if vec_env:
            obs = vec_env.normalize_obs(obs)
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        print(f"\n📍 Episode {episode + 1}/{args.episodes}")
        
        while not done:
            # 预测动作
            action, _ = model.predict(obs, deterministic=True)
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if vec_env:
                obs = vec_env.normalize_obs(obs)
            
            episode_reward += reward
            episode_length += 1
            
        
        # 记录统计
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        
        print(f"  奖励: {episode_reward:.2f}, 步数: {episode_length}")
    
    # 打印总结
    print("\n" + "=" * 60)
    print("📊 测试总结")
    print("=" * 60)
    print(f"测试回合数: {args.episodes}")
    print(f"成功率: {success_count}/{args.episodes} ({100*success_count/args.episodes:.1f}%)")
    print(f"平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"平均步数: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"最高奖励: {np.max(episode_rewards):.2f}")
    print(f"最低奖励: {np.min(episode_rewards):.2f}")
    print("=" * 60)
    
    env.close()


# ============================================================================
# 模式3: 运行模式
# ============================================================================

def run_model(args):
    """使用RL模型运行任务"""
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from isaacsim import SimulationApp
    
    print("=" * 60)
    print(f"🎮 运行模型 - {args.model}")
    print("=" * 60)
    
    # 启动Isaac Sim
    simulation_app = SimulationApp({"headless": args.headless})
    
    from isaacsim.core.api import World
    from class_taskEnv import taskEnv_SceneSetup
    from class_controller import RLController
    
    # 创建世界和任务
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    
    task = taskEnv_SceneSetup(name="env_armPick", cube_num=args.cube_num)
    world.add_task(task)
    world.reset()
    
    # 获取机器人
    task_params = task.get_params()
    robot_name = task_params["robot_name"]["value"]
    robot = world.scene.get_object(robot_name)
    
    # 加载模型
    print(f"加载模型: {args.model}")
    if args.algorithm == "ppo":
        rl_model = PPO.load(args.model)
    elif args.algorithm == "sac":
        rl_model = SAC.load(args.model)
    else:
        raise ValueError(f"不支持的算法: {args.algorithm}")
    
    # 加载VecNormalize
    vec_normalize = None
    if args.vec_normalize:
        vec_normalize = VecNormalize.load(
            args.vec_normalize,
            DummyVecEnv([lambda: None])
        )
        vec_normalize.training = False
        vec_normalize.norm_reward = False
        print(f"加载VecNormalize: {args.vec_normalize}")
    
    # 创建RL控制器
    controller = RLController(
        name="rl_controller",
        articulation=robot,
        vec_normalize=vec_normalize
    )
    controller.set_model(rl_model, vec_normalize)
    
    articulation_controller = robot.get_articulation_controller()
    
    # 任务状态
    current_cube_idx = 0
    has_grasped = False
    step_count = 0
    cube_names = task.get_cube_names()
    
    print("\n开始运行...")
    print("按 Ctrl+C 停止")
    print("-" * 60)
    
    reset_needed = False
    
    try:
        while simulation_app.is_running():
            world.step(render=True)
            
            if world.is_stopped() and not reset_needed:
                reset_needed = True
                
            if world.is_playing():
                # 获取观测
                observations = task.get_observations()
                
                # 准备RL控制器所需的观测 (修正版)
                if current_cube_idx < len(cube_names):
                    current_cube_name = cube_names[current_cube_idx]
                    cube_position = observations[current_cube_name]["position"]
                    # BUG修复: 键名是 "color_idx", 不是 "color"
                    cube_color_idx = observations[current_cube_name]["color_idx"]
                    target_position = observations["target_positions"][cube_color_idx]
                    
                    # 填充控制器需要的所有信息
                    observations["current_cube_position"] = cube_position
                    observations["current_target_position"] = target_position
                    # gripper_state 需要在循环中自己维护，就像训练时一样
                    observations["gripper_state"] = 1.0 if has_grasped else 0.0
                
                # 使用RL控制器
                actions = controller.forward(observations=observations)
                articulation_controller.apply_action(actions)
                
                step_count += 1
                
                # 更新 has_grasped 状态 (简化版逻辑)
                ee_pos, _ = robot.end_effector.get_world_pose()
                cube_pos = observations.get("current_cube_position", np.zeros(3))
                dist_ee_cube = np.linalg.norm(ee_pos - cube_pos)
                
                if not has_grasped and dist_ee_cube < 0.05 and robot.gripper.is_closed() and cube_pos[2] > 0.02:
                    has_grasped = True
                    print(">> [RUN MODE] Grasped Cube")
                
                if has_grasped and cube_pos[2] < 0.02:
                    has_grasped = False
                    print(">> [RUN MODE] Released Cube")
                # ... (rest of the loop) ...
    
    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        simulation_app.close()
        print("程序结束")


# ============================================================================
# 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RL一体化脚本 - 训练、测试、运行",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  训练模型:
    python rl_all_in_one.py train --algorithm ppo --timesteps 100000
    
  测试模型:
    python rl_all_in_one.py test --model ./models/ppo_armpickplace_final.zip --episodes 5
    
  运行模型:
    python rl_all_in_one.py run --model ./models/ppo_armpickplace_final.zip
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='选择模式')
    
    # 训练模式参数
    train_parser = subparsers.add_parser('train', help='训练模式')
    train_parser.add_argument('--algorithm', type=str, default='ppo', 
                             choices=['ppo', 'sac'], help='RL算法')
    train_parser.add_argument('--timesteps', type=int, default=100000, 
                             help='训练步数')
    train_parser.add_argument('--cube-num', type=int, default=1, 
                             help='方块数量')
    train_parser.add_argument('--headless', action='store_true', 
                             help='无头模式')
    
    # 测试模式参数
    test_parser = subparsers.add_parser('test', help='测试模式')
    test_parser.add_argument('--model', type=str, required=True, 
                            help='模型文件路径')
    test_parser.add_argument('--vec-normalize', type=str, default=None, 
                            help='VecNormalize文件路径')
    test_parser.add_argument('--algorithm', type=str, default='ppo', 
                            choices=['ppo', 'sac'], help='算法类型')
    test_parser.add_argument('--episodes', type=int, default=5, 
                            help='测试回合数')
    test_parser.add_argument('--cube-num', type=int, default=6, 
                            help='方块数量')
    test_parser.add_argument('--headless', action='store_true', 
                            help='无头模式')
    
    # 运行模式参数
    run_parser = subparsers.add_parser('run', help='运行模式')
    run_parser.add_argument('--model', type=str, required=True, 
                           help='模型文件路径')
    run_parser.add_argument('--vec-normalize', type=str, default=None, 
                           help='VecNormalize文件路径')
    run_parser.add_argument('--algorithm', type=str, default='ppo', 
                           choices=['ppo', 'sac'], help='算法类型')
    run_parser.add_argument('--cube-num', type=int, default=6, 
                           help='方块数量')
    run_parser.add_argument('--headless', action='store_true', 
                           help='无头模式')
    
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        return
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 根据模式执行
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'test':
        test_model(args)
    elif args.mode == 'run':
        run_model(args)


if __name__ == "__main__":
    main()
