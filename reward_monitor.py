#!/usr/bin/env python3
"""
奖励监控和评估脚本
帮助理解训练进度和奖励质量
"""

import matplotlib.pyplot as plt
import numpy as np

# 奖励阶段定义
reward_stages = {
    "随机探索": {
        "range": (-0.5, 0.5),
        "description": "机器人随机移动，偶尔接近方块",
        "expected_behavior": "无明确目标，奖励接近0",
        "milestones": "距离奖励很小，无里程碑",
    },
    "学习接近": {
        "range": (0.5, 5.0),
        "description": "开始学会接近方块",
        "expected_behavior": "末端执行器向方块移动",
        "milestones": "距离奖励增加，偶尔触发reach_cube (50分)",
    },
    "学习抓取": {
        "range": (5.0, 50.0),
        "description": "频繁接近并尝试抓取",
        "expected_behavior": "能稳定接近，开始触发抓取",
        "milestones": "reach_cube常触发，grasp_success (200分) 开始出现",
    },
    "稳定抓取": {
        "range": (50.0, 200.0),
        "description": "稳定抓取并尝试搬运",
        "expected_behavior": "能抓起方块并移动",
        "milestones": "grasp_success频繁触发，向目标移动",
    },
    "学习放置": {
        "range": (200.0, 500.0),
        "description": "学习将方块放到目标位置",
        "expected_behavior": "抓起后向目标移动",
        "milestones": "开始触发place_success (500分)",
    },
    "熟练掌握": {
        "range": (500.0, 1000.0),
        "description": "能高效完成整个任务",
        "expected_behavior": "快速完成抓取-搬运-放置",
        "milestones": "所有里程碑稳定触发，距离奖励高",
    },
    "专家级别": {
        "range": (1000.0, float('inf')),
        "description": "接近最优策略",
        "expected_behavior": "最短路径完成任务",
        "milestones": "高效率，低动作惩罚",
    },
}

def evaluate_reward(mean_reward):
    """根据平均奖励评估训练状态"""
    print("\n" + "="*70)
    print("📊 奖励评估报告")
    print("="*70)
    print(f"\n当前平均奖励: {mean_reward:.2f}")
    
    # 找到对应的阶段
    current_stage = None
    for stage_name, info in reward_stages.items():
        if info["range"][0] <= mean_reward < info["range"][1]:
            current_stage = stage_name
            break
    
    if current_stage is None and mean_reward >= 1000.0:
        current_stage = "专家级别"
    
    if current_stage:
        info = reward_stages[current_stage]
        print(f"\n🎯 当前阶段: {current_stage}")
        print(f"   奖励范围: {info['range'][0]:.1f} ~ {info['range'][1]:.1f}")
        print(f"   阶段描述: {info['description']}")
        print(f"   预期行为: {info['expected_behavior']}")
        print(f"   里程碑: {info['milestones']}")
        
        # 下一阶段
        stage_names = list(reward_stages.keys())
        current_idx = stage_names.index(current_stage)
        if current_idx < len(stage_names) - 1:
            next_stage = stage_names[current_idx + 1]
            next_info = reward_stages[next_stage]
            print(f"\n⏭️  下一阶段: {next_stage}")
            print(f"   目标奖励: {next_info['range'][0]:.1f}+")
            print(f"   需要实现: {next_info['expected_behavior']}")
    else:
        print("\n⚠️  奖励异常低，可能需要检查环境或奖励函数")
    
    print("\n" + "="*70)

def plot_reward_stages():
    """可视化奖励阶段"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 绘制阶段区域
    colors = ['#ff9999', '#ffcc99', '#ffff99', '#99ff99', '#99ccff', '#cc99ff', '#ff99ff']
    stage_names = list(reward_stages.keys())
    
    for i, (stage_name, info) in enumerate(reward_stages.items()):
        if info["range"][1] == float('inf'):
            continue
        y_pos = i
        width = info["range"][1] - info["range"][0]
        ax.barh(y_pos, width, left=info["range"][0], height=0.8, 
                color=colors[i], alpha=0.6, edgecolor='black')
        
        # 添加标签
        mid_point = (info["range"][0] + info["range"][1]) / 2
        ax.text(mid_point, y_pos, f"{stage_name}\n{info['range'][0]:.0f}-{info['range'][1]:.0f}", 
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 添加里程碑标记
    milestones = {
        50: "reach_cube",
        200: "grasp_success", 
        500: "place_success",
    }
    
    for reward_val, name in milestones.items():
        ax.axvline(x=reward_val, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.text(reward_val, len(stage_names)-0.5, name, 
                rotation=90, va='bottom', ha='right', fontsize=9, color='red')
    
    ax.set_yticks(range(len(stage_names)))
    ax.set_yticklabels(stage_names)
    ax.set_xlabel('Episode Reward', fontsize=12)
    ax.set_title('训练阶段与奖励对应关系', fontsize=14, fontweight='bold')
    ax.set_xlim(-1, 1200)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reward_stages.png', dpi=150, bbox_inches='tight')
    print("📈 图表已保存: reward_stages.png")
    plt.show()

def analyze_tensorboard_log(log_file=None):
    """分析TensorBoard日志中的奖励趋势"""
    print("\n💡 提示: 使用TensorBoard查看详细训练曲线")
    print("   命令: tensorboard --logdir=logs")
    print("\n关注的关键指标:")
    print("   1. rollout/mean_reward - 平均奖励趋势")
    print("   2. rollout/mean_value - 价值网络预测（应接近实际奖励）")
    print("   3. train/explained_variance - 应该>0.5（越接近1越好）")
    print("   4. train/policy_gradient_loss - 梯度大小")
    print("\n健康训练的信号:")
    print("   ✅ mean_reward 稳定上升")
    print("   ✅ explained_variance > 0.7")
    print("   ✅ 无频繁的突然下降")
    print("   ✅ mean_value 跟随 mean_reward 增长")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="奖励监控工具")
    parser.add_argument("--reward", type=float, help="当前平均奖励值")
    parser.add_argument("--plot", action="store_true", help="生成奖励阶段可视化图")
    args = parser.parse_args()
    
    print("="*70)
    print("🎯 Franka Pick-Place 奖励评估系统")
    print("="*70)
    
    if args.plot:
        plot_reward_stages()
    
    if args.reward is not None:
        evaluate_reward(args.reward)
    else:
        print("\n📚 奖励组成说明:")
        print("\n1. 距离奖励 (持续):")
        print("   - 接近方块: exp(-5 × distance) × 2.0")
        print("   - 接近目标: exp(-3 × distance) × 3.0")
        print("\n2. 里程碑奖励 (一次性):")
        print("   - 接近方块 (<8cm): +50")
        print("   - 抓取成功 (抬起): +200")
        print("   - 放置成功: +500")
        print("\n3. 动作惩罚:")
        print("   - -0.01 × Σ(action²)")
        print("\n理论最大值: ~1300-1650 / episode")
        print("\n使用方法:")
        print("  python reward_monitor.py --reward 0.067  # 评估当前奖励")
        print("  python reward_monitor.py --plot          # 生成可视化图")
    
    analyze_tensorboard_log()
