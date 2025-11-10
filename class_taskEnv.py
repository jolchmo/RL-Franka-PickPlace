# class_taskEnv.py
# 这是一个完整、修正后的版本，整合了所有讨论过的修复和改进。
#
# 主要改进:
# 1. 重写了奖励函数(_compute_reward)，使其基于真实环境状态并提供密集的、分阶段的奖励。
# 2. 修正了step函数中的动作执行方式，改用更稳定、更高级的末端执行器目标位姿控制。
# 3. 奖励函数现在接收action作为参数，实现了更符合理论的 R(s, a)。
# 4. 增加了详细的注释，解释了关键部分的设计思想。

import numpy as np
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.scenes import Scene
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.types import ArticulationAction

# 强化学习相关导入
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYMNASIUM_AVAILABLE = True
except ImportError:
    print("警告: gymnasium 未安装。RL 环境将不可用。")
    print("请运行: pip install gymnasium")
    GYMNASIUM_AVAILABLE = False


class taskEnv_SceneSetup(BaseTask):
    """
    负责场景设置的基础任务类。
    这个类主要用于在Isaac Sim中初始化场景，包括机器人和方块。
    """

    def __init__(self, name="task_arm_pickplace", cube_num=8, cube_scale=None):
        super().__init__(name, offset=None)
        self._robot = None
        self._cubes = []
        self._cube_positions = []
        self._cube_colors_rgb = []
        self._cube_colors_idx = []

        self._cube_num = cube_num
        self._stage_units = get_stage_units()

        if cube_scale is None:
            self._cube_scale = np.array([0.05, 0.05, 0.05])
        else:
            self._cube_scale = cube_scale

        # 目标位置和颜色定义 (单位: 米)
        self._target_positions_meters = np.array([
            [0.4, 0.3, 0.0],   # Red
            [0.4, 0.0, 0.0],   # Green
            [0.4, -0.3, 0.0]   # Blue
        ])
        self._target_colors = np.array([
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0]   # Blue
        ])

        # 将目标位置转换为场景单位
        self._target_positions = self._target_positions_meters / self._stage_units
        self._task_objects = {}

    def set_up_scene(self, scene: Scene) -> None:
        super().set_up_scene(scene)

        # 添加Franka机械臂
        # 确保在使用RL环境前机器人已被加载
        try:
            from isaacsim.robot.manipulators.examples.franka import Franka
            self._robot = scene.add(Franka(prim_path="/World/Franka", name="my_franka"))
            self._task_objects[self._robot.name] = self._robot
        except Exception as e:
            raise RuntimeError(f"加载Franka机械臂失败: {e}。请确保Franka扩展已启用。")

        # 添加随机方块
        for _ in range(self._cube_num):
            cube_prim_path = find_unique_string_name("/World/Cube", lambda x: not is_prim_path_valid(x))
            cube_name = find_unique_string_name("cube", lambda x: not scene.object_exists(x))

            position, orientation, color_rgb, color_idx = self._get_random_cube_pose()

            cube = scene.add(
                DynamicCuboid(
                    name=cube_name,
                    prim_path=cube_prim_path,
                    position=position,
                    orientation=orientation,
                    scale=self._cube_scale / self._stage_units,
                    size=1.0,
                    color=color_rgb,
                    mass=0.05,  # 给方块一个合理的质量
                )
            )
            self._cubes.append(cube)
            self._task_objects[cube.name] = cube

    def _get_random_cube_pose(self):
        # 在一个固定区域内随机生成方块位置
        # 位置单位: 场景单位
        x = np.random.uniform(0.1, 0.3) / self._stage_units
        y = np.random.uniform(-0.3, 0.3) / self._stage_units
        z = (self._cube_scale[2] / 2.0) / self._stage_units
        position = np.array([x, y, z])

        # 检查重叠
        while any(np.linalg.norm(position - p) < self._cube_scale[0] * 2 / self._stage_units for p in self._cube_positions):
            x = np.random.uniform(0.1, 0.3) / self._stage_units
            y = np.random.uniform(-0.3, 0.3) / self._stage_units
            position = np.array([x, y, z])

        self._cube_positions.append(position)

        # 随机选择颜色
        color_idx = np.random.randint(0, len(self._target_colors))
        color_rgb = self._target_colors[color_idx]
        self._cube_colors_rgb.append(color_rgb)
        self._cube_colors_idx.append(color_idx)

        return position, None, color_rgb, color_idx

    def get_observations(self) -> dict:
        # print(f"self._cubes,{len(self._cubes)}")
        # print(f"self._cubes_color,{len(self._cube_colors_idx)}")
        observations = {}
        if self._robot:
            observations[self._robot.name] = {
                "joint_positions": self._robot.get_joint_positions(),
                "end_effector_position": self._robot.end_effector.get_world_pose()[0],
            }

        for i, cube in enumerate(self._cubes):
            pos, ori = cube.get_world_pose()
            observations[cube.name] = {
                "position": pos,
                "orientation": ori,
                "color_idx": self._cube_colors_idx[i]
            }

        observations["target_positions"] = self._target_positions
        return observations

    def get_params(self) -> dict:
        return {"robot_name": {"value": self._robot.name, "modifiable": False}}

    def get_cube_names(self):
        return [cube.name for cube in self._cubes]

    def reset(self):
        """重置方块位置和颜色"""
        if hasattr(self, '_cubes') and self._cubes:
            # 清空之前的记录
            self._cube_positions = []
            self._cube_colors_rgb = []
            self._cube_colors_idx = []
            
            # 为每个方块重新生成位置和颜色
            for cube in self._cubes:
                position, orientation, color_rgb, color_idx = self._get_random_cube_pose()
                
                # 更新方块的物理位置
                cube.set_world_pose(position=position)
                cube.set_default_state(position=position)
                # 注意：DynamicCuboid的颜色在创建后无法动态修改
                # 所以颜色只在内部逻辑中更新，视觉上保持原色
        return True

    def cleanup(self) -> None:
        self._cube = [] 
        # 在重置或关闭时清空列表，确保状态正确
        self._cube_positions = []
        self._cube_colors_rgb = []
        self._cube_colors_idx = []
        super().cleanup()

# class_taskEnv.py
# ... (保留文件顶部的其他import) ...
# 注意：我们不再需要 IKSolver 或任何 Omnigraph 的东西了
# ... (保留 taskEnv_SceneSetup 类的定义) ...
# ==============================================================================
#  ↓↓↓ 将下面这个完整的类替换掉您文件中旧的 ArmPickPlaceRLEnv 类 ↓↓↓
# ==============================================================================
class ArmPickPlaceRLEnv(gym.Env):
    """
    【关节空间控制版】
    RL Agent 直接输出每个关节的目标位置增量和夹爪指令。
    这种方式更底层、更直接，但对 Agent 的学习能力要求更高。
    """
    metadata = {"render_modes": ["human", "rgb_array"]}
    def __init__(self, headless=False, render_mode=None, max_episode_steps=750, cube_num=8, simulation_app=None):
        if not GYMNASIUM_AVAILABLE:
            raise ImportError("gymnasium is required. Please run: pip install gymnasium")
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.cube_num = cube_num
        
        if simulation_app is None:
            from isaacsim import SimulationApp
            self.simulation_app = SimulationApp({"headless": headless})
            self._owns_simulation_app = True
        else:
            self.simulation_app = simulation_app
            self._owns_simulation_app = False
        from isaacsim.core.api import World
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        self.task = taskEnv_SceneSetup(name="env_armPick", cube_num=self.cube_num)
        self.world.add_task(self.task)
        self.world.reset()
        task_params = self.task.get_params()
        self.robot_name = task_params["robot_name"]["value"]
        self.robot = self.world.scene.get_object(self.robot_name)
        self.articulation_controller = self.robot.get_articulation_controller()
        
        # 【核心修改 1】: 定义关节空间下的动作空间
        # Franka 有7个臂部关节 + 1个用于控制夹爪开合的动作
        num_arm_joints = 7
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_arm_joints + 1,), dtype=np.float32)
        
        # 观测空间保持不变，因为它已经包含了关节位置和任务所需信息
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32)
        # 任务相关的逻辑状态变量
        self.current_step = 0
        self.cube_names = []
        self.current_cube_idx = 0
        self.has_grasped = False
        self.last_distance_to_cube = None
        self.last_distance_to_target = None
    def step(self, action: np.ndarray):
        """
        【关节空间控制版】的 step 函数
        """
        self.current_step += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # 【核心修改 2】: 将动作解析为关节指令
        
        # 1. 提取关节动作和夹爪动作
        # 前7个值用于控制7个臂部关节
        joint_actions = action[:7]
        # 最后1个值用于控制夹爪
        gripper_action = action[7]
        # 2. 计算目标关节位置（增量式控制）
        # 这是标准的做法：Agent输出的是一个小的“变化量”，而不是绝对位置
        action_scale = 0.05  # 每个step关节角度的最大变化量（单位：弧度）
        
        # 获取当前所有关节的位置（Franka有9个自由度：7臂+2指）
        current_joint_positions = self.robot.get_joint_positions()
        
        # 只修改臂部关节（前7个）
        # 创建一个新的目标位置数组，防止修改原始数据
        target_joint_positions = np.copy(current_joint_positions)
        target_joint_positions[:7] += joint_actions * action_scale
        
        # 3. 应用关节动作
        # ArticulationAction可以直接接收关节位置指令
        robot_action = ArticulationAction(joint_positions=target_joint_positions)
        self.articulation_controller.apply_action(robot_action)
        # 4. 夹爪控制逻辑保持不变
        if gripper_action > 0.5:
            self.robot.gripper.close()
        elif gripper_action < -0.5:
            self.robot.gripper.open()
        # 推进仿真并获取结果
        if self.render_mode == "haman":
            self.world.step(render=True)
        else:
            self.world.step(render=False)
        obs = self._get_obs()
        reward, terminated = self._compute_reward(action)
        truncated = self.current_step >= self.max_episode_steps
        info = {"is_success": self.current_cube_idx >= len(self.cube_names)}
        return obs, reward, terminated, truncated, info
    def reset(self, seed=None, options=None):
        # 这个函数基本不变，只是不再需要重置IK求解器了
        super().reset(seed=seed)
        self.world.reset()
        self.task.reset()
        self.current_step = 0
        self.cube_names = self.task.get_cube_names()
        self.current_cube_idx, self.has_grasped = 0, False
        self.last_distance_to_cube, self.last_distance_to_target = None, None
        
        # 重置到初始关节位置并打开夹爪
        self.robot.gripper.open()
        # 等待几步让夹爪动作完成
        for _ in range(5):
            self.world.step()
        
        obs, info = self._get_obs(), {}
        return obs, info
    # _get_obs, _compute_reward, close 函数可以保持完全不变
    # ... (此处省略这些未改变的函数，直接使用您文件中的版本即可)
    def _get_obs(self):
        obs_dict = self.task.get_observations()
        ee_pos, _ = self.robot.end_effector.get_world_pose()
        joint_pos = obs_dict[self.robot_name]["joint_positions"]
        if self.current_cube_idx < len(self.cube_names):
            cube_name = self.cube_names[self.current_cube_idx]
            cube_pos = obs_dict[cube_name]["position"]
            color_idx = obs_dict[cube_name]["color_idx"]
            target_pos = obs_dict["target_positions"][color_idx]
        else:
            cube_pos, color_idx, target_pos = np.zeros(3), 0, np.zeros(3)
        relative_vec = (cube_pos - ee_pos) if not self.has_grasped else (target_pos - cube_pos)
        distance = np.linalg.norm(relative_vec)
        gripper_state = np.array([1.0 if self.has_grasped else 0.0])
        return np.concatenate([
            ee_pos.astype(np.float32), joint_pos.astype(np.float32), relative_vec.astype(np.float32),
            np.array([distance], dtype=np.float32), cube_pos.astype(np.float32),
            target_pos.astype(np.float32), gripper_state
        ])
    def _compute_reward(self, action: np.ndarray):
        reward = 0.0
        obs_dict = self.task.get_observations()
        ee_pos, _ = self.robot.end_effector.get_world_pose()
        if self.current_cube_idx >= len(self.cube_names): return 500.0, True
        cube_name = self.cube_names[self.current_cube_idx]
        cube_pos, color_idx = obs_dict[cube_name]["position"], obs_dict[cube_name]["color_idx"]
        target_pos = obs_dict["target_positions"][color_idx]
        dist_ee_to_cube = np.linalg.norm(ee_pos - cube_pos)
        dist_cube_to_target = np.linalg.norm(cube_pos - target_pos)
        if not self.has_grasped:
            if self.last_distance_to_cube is not None: reward += (self.last_distance_to_cube - dist_ee_to_cube) * 10.0
            if dist_ee_to_cube < 0.05:
                reward += 10.0
                if action[7] > 0.5: # 夹爪动作在索引7
                    reward += 10.0
                    if cube_pos[2] > 0.03: self.has_grasped = True; reward += 200.0
        else:
            if self.last_distance_to_target is not None: reward += (self.last_distance_to_target - dist_cube_to_target) * 15.0
            if dist_cube_to_target < 0.05:
                reward += 50.0
                if action[7] < -0.5: # 夹爪动作在索引7
                    reward += 30.0
                    if cube_pos[2] < 0.05:
                        reward += 250.0; self.has_grasped = False; self.current_cube_idx += 1
                        if self.current_cube_idx >= len(self.cube_names): return reward + 500.0, True
        self.last_distance_to_cube = dist_ee_to_cube
        if self.has_grasped: self.last_distance_to_target = dist_cube_to_target
        if ee_pos[2] > 0.8: reward -= 5.0
        elif ee_pos[2] < 0.0: reward -= 10.0
        return reward, False
    def close(self):
        if self._owns_simulation_app and hasattr(self, 'simulation_app'):
            self.simulation_app.close()