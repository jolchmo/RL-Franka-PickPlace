# RL一体化脚本使用指南

## 📝 简介

`rl_all_in_one.py` 是一个集训练、测试、运行于一体的脚本，提供三种模式：

1. **训练模式 (train)**: 训练新的RL模型
2. **测试模式 (test)**: 评估训练好的模型性能
3. **运行模式 (run)**: 在Isaac Sim中实时运行模型

## 🚀 快速开始

### 1️⃣ 训练模式

训练一个新的PPO模型：

```powershell
C:\issac-sim\python.bat rl_all_in_one.py train --algorithm ppo --timesteps 100000
```

```
C:\issac-sim\python.bat rl_all_in_one.py train `
  --algorithm ppo `
  --timesteps 200000 `
  --cube-num 2 `
  --headless `
  --resume-from models/ppo_armpickplace_final.zip `
  --resume-vecnormalize models/vec_normalize_ppo.pkl
```
 

**参数说明:**
- `--algorithm`: 算法类型 (`ppo` 或 `sac`)
- `--timesteps`: 训练步数 (默认: 100000)
- `--cube-num`: 方块数量 (默认: 6)
- `--headless`: 无头模式运行 (不显示GUI，训练更快)


### 2️⃣ 测试模式

测试训练好的模型性能：

```powershell
C:\issac-sim\python.bat rl_all_in_one.py test --model .\models\ppo_armpickplace_final.zip --episodes 5 --cube-num 1
```

**参数说明:**
- `--model`: 模型文件路径 (必需)
- `--vec-normalize`: VecNormalize文件路径 (可选)
- `--algorithm`: 算法类型 (默认: ppo)
- `--episodes`: 测试回合数 (默认: 5)
- `--cube-num`: 方块数量 (默认: 6)
- `--headless`: 无头模式

**示例:**

```powershell
# 完整测试（带VecNormalize）
python rl_all_in_one.py test ^
    --model ./models/ppo_armpickplace_final.zip ^
    --vec-normalize ./models/vec_normalize_ppo.pkl ^
    --algorithm ppo ^
    --episodes 10

python rl_all_in_one.py test --model ./models/ppo_armpickplace_final.zip --vec-normalize ./models/vec_normalize_ppo.pkl --algorithm ppo


# 快速测试（无头模式）
python rl_all_in_one.py test ^
    --model ./models/ppo_armpickplace_final.zip ^
    --vec-normalize ./models/vec_normalize_ppo.pkl ^
    --headless ^
    --episodes 3

# 测试SAC模型
python rl_all_in_one.py test ^
    --model ./models/sac_armpickplace_final.zip ^
    --vec-normalize ./models/vec_normalize_sac.pkl ^
    --algorithm sac ^
    --episodes 5
```

**输出信息:**
- 每个episode的奖励和步数
- 成功率统计
- 平均奖励和标准差
- 平均步数

---

### 3️⃣ 运行模式

在Isaac Sim中实时运行模型：

```powershell
python rl_all_in_one.py run --model ./models/ppo_armpickplace_final.zip
```

**参数说明:**
- `--model`: 模型文件路径 (必需)
- `--vec-normalize`: VecNormalize文件路径 (可选但推荐)
- `--algorithm`: 算法类型 (默认: ppo)
- `--cube-num`: 方块数量 (默认: 6)
- `--headless`: 无头模式

**示例:**

```powershell
# 完整运行（带VecNormalize）
python rl_all_in_one.py run ^
    --model ./models/ppo_armpickplace_final.zip ^
    --vec-normalize ./models/vec_normalize_ppo.pkl ^
    --algorithm ppo

# 改变场景设置
python rl_all_in_one.py run ^
    --model ./models/ppo_armpickplace_final.zip ^
    --vec-normalize ./models/vec_normalize_ppo.pkl ^
    --cube-num 8

# 运行SAC模型
python rl_all_in_one.py run ^
    --model ./models/sac_armpickplace_final.zip ^
    --vec-normalize ./models/vec_normalize_sac.pkl ^
    --algorithm sac
```

**运行中:**
- 实时显示机械臂状态
- 每100步打印一次进度
- 任务完成后自动重置
- 按 `Ctrl+C` 退出

---

## 📊 完整工作流程

### 从零开始的完整流程：

```powershell
# 步骤1: 训练模型
python rl_all_in_one.py train --algorithm ppo --timesteps 100000 --headless

# 步骤2: 测试性能
python rl_all_in_one.py test ^
    --model ./models/ppo_armpickplace_final.zip ^
    --vec-normalize ./models/vec_normalize_ppo.pkl ^
    --episodes 5

# 步骤3: 实际运行
python rl_all_in_one.py run ^
    --model ./models/ppo_armpickplace_final.zip ^
    --vec-normalize ./models/vec_normalize_ppo.pkl
```

---

## ⚙️ 高级用法

### 继续训练

如果想在现有模型基础上继续训练，可以：

1. 先加载模型
2. 修改脚本中的 `train_model` 函数
3. 使用 `model.load()` 加载现有模型

### 调整超参数

在 `rl_all_in_one.py` 中找到对应算法的配置部分：

```python
# PPO配置
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,    # 学习率
    n_steps=2048,          # 收集步数
    batch_size=64,         # 批大小
    n_epochs=10,           # 训练轮数
    # ... 更多参数
)
```

### 对比不同算法

```powershell
# 训练PPO
python rl_all_in_one.py train --algorithm ppo --timesteps 100000 --headless

# 训练SAC
python rl_all_in_one.py train --algorithm sac --timesteps 100000 --headless

# 测试对比
python rl_all_in_one.py test --model ./models/ppo_armpickplace_final.zip --episodes 10
python rl_all_in_one.py test --model ./models/sac_armpickplace_final.zip --episodes 10
```

---

## 📁 文件结构

```
task_armPickPlace/
├── rl_all_in_one.py          ← 一体化脚本 (主文件)
├── class_taskEnv.py           ← 任务环境和RL环境
├── class_controller.py        ← 控制器
├── requirements_rl.txt        ← 依赖包
├── models/                    ← 保存的模型
│   ├── ppo_armpickplace_final.zip
│   ├── vec_normalize_ppo.pkl
│   └── ...checkpoints
└── logs/                      ← TensorBoard日志
    └── PPO_1/
```

---

## 🎯 使用建议

### 训练建议

1. **初次训练**: 使用较少方块 (3-4个) 和较少步数 (50K)
2. **使用无头模式**: 加快训练速度 (2-3倍)
3. **监控训练**: 使用TensorBoard实时查看奖励曲线
4. **保存checkpoint**: 每10K步自动保存，避免意外中断

### 测试建议

1. **多次测试**: 至少测试5-10个回合获取统计信息
2. **记录性能**: 比较不同模型的成功率和效率
3. **调整难度**: 逐步增加方块数量测试泛化能力

### 运行建议

1. **使用VecNormalize**: 确保运行时使用训练时的标准化
2. **观察行为**: 注意模型的抓取和放置策略
3. **记录问题**: 如果模型表现不佳，记录失败case用于改进

---

## ❓ 常见问题

### Q1: 训练很慢怎么办？
**A:** 使用 `--headless` 参数开启无头模式，可以快2-3倍。

### Q2: 如何知道训练是否收敛？
**A:** 使用 `tensorboard --logdir ./logs/` 查看奖励曲线，如果曲线趋于平稳就说明收敛了。

### Q3: 测试成功率低怎么办？
**A:** 
- 增加训练步数
- 降低任务难度（减少方块数量）
- 调整奖励函数权重
- 尝试不同的算法（PPO vs SAC）

### Q4: 运行时模型表现和测试不一样？
**A:** 确保使用相同的 `--vec-normalize` 参数加载标准化文件。

### Q5: 可以在训练时更改参数吗？
**A:** 不建议。如果需要调整，应该重新训练或使用继续训练模式。

---

## 📚 参考资料

- **Stable-Baselines3文档**: https://stable-baselines3.readthedocs.io/
- **Gymnasium文档**: https://gymnasium.farama.org/
- **Isaac Sim文档**: https://docs.omniverse.nvidia.com/isaacsim/

---

## 🎉 总结

`rl_all_in_one.py` 提供了一个简单易用的接口来完成RL的完整流程：

```
训练 (train) → 测试 (test) → 运行 (run)
```

所有功能都在一个文件中，无需切换多个脚本！

**开始使用:**
```powershell
python rl_all_in_one.py train --algorithm ppo --timesteps 100000 --headless
```

祝训练顺利！🚀
