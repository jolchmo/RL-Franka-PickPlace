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
            from isaacsim.robot.manipulators.franka import Franka
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

    def cleanup(self) -> None:
        # 在重置或关闭时清空列表，确保状态正确
        self._cube_positions = []
        self._cube_colors_rgb = []
        self._cube_colors_idx = []
        super().cleanup()


class ArmPickPlaceRLEnv(gym.Env):
    """
    将taskEnv_SceneSetup包装为符合Gymnasium接口的强化学习环境。
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, headless=False, render_mode=None, max_episode_steps=750, cube_num=6, simulation_app=None):
        if not GYMNASIUM_AVAILABLE:
            raise ImportError("gymnasium is required. Please run: pip install gymnasium")

        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.cube_num = cube_num

        # 定义观测空间 (20维)
        # ee_pos(3), joint_pos(9), cube_pos(3), cube_color(1), target_pos(3), gripper_state(1)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)

        # 定义动作空间 (4维)
        # delta_x,y,z: 末端位置增量 [-1, 1], gripper_action: -1(开) to 1(合)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

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

        # 任务相关的逻辑状态变量
        self.current_step = 0
        self.cube_names = []
        self.current_cube_idx = 0
        self.has_grasped = False
        self.last_distance_to_cube = None
        self.last_distance_to_target = None

    def _get_obs(self):
        """获取当前观测，并转换为扁平化的numpy数组。"""
        obs_dict = self.task.get_observations()
        ee_pos, _ = self.robot.end_effector.get_world_pose()
        joint_pos = obs_dict[self.robot_name]["joint_positions"]

        if self.current_cube_idx < len(self.cube_names):
            cube_name = self.cube_names[self.current_cube_idx]
            cube_pos = obs_dict[cube_name]["position"]
            color_idx = obs_dict[cube_name]["color_idx"]
            target_pos = obs_dict["target_positions"][color_idx]
        else:  # 所有方块都处理完成
            cube_pos = np.zeros(3)
            color_idx = 0
            target_pos = np.zeros(3)

        # 夹爪状态：1.0表示已抓取物体，0.0表示未抓取
        gripper_state = np.array([1.0 if self.has_grasped else 0.0])

        obs_array = np.concatenate([
            ee_pos.astype(np.float32),
            joint_pos.astype(np.float32),
            cube_pos.astype(np.float32),
            np.array([color_idx], dtype=np.float32),
            target_pos.astype(np.float32),
            gripper_state
        ])
        return obs_array

    def step(self, action: np.ndarray):
        """执行一步动作，并返回(obs, reward, terminated, truncated, info)"""
        self.current_step += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # --- 1. 应用动作 ---
        # 使用末端执行器控制器，这是更稳定、更推荐的方式
        current_ee_pos, current_ee_ori = self.robot.end_effector.get_world_pose()
        action_scale = 0.05  # 每步最大移动0.05米
        target_ee_pos = current_ee_pos + action[:3] * action_scale

        self.robot.end_effector.apply_action(
            target_positions=target_ee_pos.reshape(1, 3),  # 必须是 (N, 3) 格式
            target_orientations=current_ee_ori.reshape(1, 4)  # 保持当前姿态
        )

        # 控制夹爪
        if action[3] > 0.5:
            self.robot.gripper.close()
        elif action[3] < -0.5:
            self.robot.gripper.open()

        # --- 2. 推进仿真 ---
        # 即使在无头模式下，render=True对于某些传感器数据的更新也可能是必要的
        self.world.step(render=self.render_mode is not None)

        # --- 3. 获取结果 ---
        obs = self._get_obs()
        reward, terminated = self._compute_reward(action)

        truncated = self.current_step >= self.max_episode_steps
        if truncated:
            reward -= 50.0  # 超时惩罚
            terminated = True  # 在Gymnasium中, 超时也认为是terminated

        info = {"is_success": self.current_cube_idx >= len(self.cube_names)}

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, action: np.ndarray):
        """
        计算奖励 R(s, a)。
        s (状态) 通过 self 访问, a (动作) 作为参数传入。
        """
        reward = 0.0

        # --- 1. 获取当前状态 ---
        obs_dict = self.task.get_observations()
        ee_pos, _ = self.robot.end_effector.get_world_pose()

        if self.current_cube_idx >= len(self.cube_names):
            return 200.0, True  # 任务完成，给予最终大奖

        cube_name = self.cube_names[self.current_cube_idx]
        cube_pos = obs_dict[cube_name]["position"]
        color_idx = obs_dict[cube_name]["color_idx"]
        target_pos = obs_dict["target_positions"][color_idx]

        dist_ee_to_cube = np.linalg.norm(ee_pos - cube_pos)
        dist_cube_to_target = np.linalg.norm(cube_pos - target_pos)

        # --- 2. 分阶段奖励设计 ---
        # 阶段一：引导夹爪接近方块 (Reaching)
        if not self.has_grasped:
            reward_reach = 1.0 - np.tanh(10.0 * dist_ee_to_cube)
            reward += reward_reach
            if self.last_distance_to_cube is not None and dist_ee_to_cube < self.last_distance_to_cube:
                reward += 0.5  # 奖励正在靠近的行为

        # 阶段二：抓取与抬起 (Grasping & Lifting)
        # 通过判断方块高度和与夹爪的距离来确定是否成功抬起
        if not self.has_grasped and cube_pos[2] > 0.03 and dist_ee_to_cube < 0.06:
            self.has_grasped = True
            reward += 30.0
            print(f"✅ (Step {self.current_step}) 方块 {self.current_cube_idx+1} 已抓起!")

        # 阶段三：引导移动到目标点 (Moving)
        if self.has_grasped:
            reward_move = (1.0 - np.tanh(10.0 * dist_cube_to_target)) * 5.0
            reward += reward_move
            if self.last_distance_to_target is not None and dist_cube_to_target < self.last_distance_to_target:
                reward += 1.0  # 奖励正在靠近目标的行为

        # 阶段四：成功放置 (Placing)
        if self.has_grasped and dist_cube_to_target < 0.08 and cube_pos[2] < 0.05:
            reward += 100.0
            print(f"🎉 (Step {self.current_step}) 方块 {self.current_cube_idx+1} 已成功放置!")
            self.has_grasped = False
            self.current_cube_idx += 1
            if self.current_cube_idx >= len(self.cube_names):
                return reward, True  # 任务完成
            self.last_distance_to_cube = None
            self.last_distance_to_target = None

        # --- 3. 更新状态用于下次计算 ---
        self.last_distance_to_cube = dist_ee_to_cube
        if self.has_grasped:
            self.last_distance_to_target = dist_cube_to_target

        # --- 4. 动作惩罚 (鼓励平滑) ---
        action_penalty = -0.01 * np.sum(np.square(action[:3]))
        reward += action_penalty

        return reward, False

    def reset(self, seed=None, options=None):
        """重置环境到初始状态"""
        super().reset(seed=seed)

        self.world.reset()

        # 重置所有逻辑状态变量
        self.current_step = 0
        self.cube_names = self.task.get_cube_names()
        self.current_cube_idx = 0
        self.has_grasped = False
        self.last_distance_to_cube = None
        self.last_distance_to_target = None

        self.robot.gripper.open()  # 确保夹爪初始是张开的

        obs = self._get_obs()
        info = {}
        return obs, info

    def render(self):
        """渲染由Isaac Sim自动处理"""
        pass

    def close(self):
        """关闭环境"""
        if self._owns_simulation_app and hasattr(self, 'simulation_app'):
            self.simulation_app.close()
