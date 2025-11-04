# Task类核心内容，通过继承BaseTask类来组织任务逻辑，实现机械臂抓取与分类任务
# 可通过修改以下内容来实现不同的任务：
# （1）场景初始化（set_up_scene）
#     负责定义任务的吃初始场景。如加载资源和配置物理参数
# （2）观察获取（get_observations）
#     负责获取任务执行过程中的观察数据，如机械臂末端位置、方块位置等
# （3）指标计算（calculate_metrics）
#     定义任务
# （4）重置（reset）
import argparse
import sys

from isaacsim.core.api.tasks import BaseTask

from isaacsim.core.api.objects import GroundPlane, DynamicCuboid
from isaacsim.core.api.scenes import Scene

import numpy as np
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.prims import is_prim_path_valid

import isaacsim.core.utils.stage as stage_utils

# 强化学习相关导入（仅在需要时导入）
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYMNASIUM_AVAILABLE = True
except ImportError:
    print("Warning: gymnasium not installed. RL environment will not be available.")
    GYMNASIUM_AVAILABLE = False


class taskEnv_SceneSetup(BaseTask):
    """机械臂抓取与分类任务"""

    def __init__(
        self,
        name="task_arm_pickplace",
        cube_num=8,
        cube_scale=None,
    ) -> None:
        super().__init__(name, offset=None)
        self._robot = None
        self._cubes = []            # 立方体列表
        self._cube_positions = []
        self._cube_colors = []

        self._cube_num = cube_num
        self._cube_scale = cube_scale
        if self._cube_scale is None:
            self._cube_scale = np.array([0.0515, 0.0515, 0.0515])/get_stage_units()  # 默认方块尺寸
        self._target_positions = np.array([[0.2, -0.2, 0.0],     # Red  分类目标位置 (3个目标位置对应3种颜色)
                                           [0.4, -0.2, 0.0],     # Green
                                           [0.6, -0.2, 0.0]])    # Blue
        self._target_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        self._target_colors = np.array([[1, 0, 0],   # Red
                                        [0, 1, 0],   # Green
                                        [0, 0, 1]])  # Blue

        # Initialize missing attributes
        self._task_objects = {}

    # （1）场景初始化
    def set_up_scene(self, scene: Scene) -> None:
        super().set_up_scene(scene)

        # 添加随机方块
        for i in range(self._cube_num):
            cube_position, cube_orientation, cube_color = self.add_random_cube()
            cube_prim_path = find_unique_string_name(
                initial_name="/World/Cube",
                is_unique_fn=lambda x: not is_prim_path_valid(x)
            )
            cube_name = find_unique_string_name(
                initial_name="cube",
                is_unique_fn=lambda x: not scene.object_exists(x)
            )
            # 创建方块对象添加到_cubes列表中，便于后续统一管理和访问
            self._cubes.append(
                scene.add(  # 将创建的DynamicCuboid对象添加到物理场景中，使得立方体参与物理仿真，返回一个场景对象的引用
                    DynamicCuboid(  # 创建具有物理属性的立方体（具有质量、惯性等物理属性），可以参与碰撞检测，能够被机器人抓取和移动
                        name=cube_name,
                        prim_path=cube_prim_path,
                        position=cube_position,
                        orientation=cube_orientation,
                        scale=self._cube_scale,
                        size=1.0,    # 立方体的边长
                        color=cube_color,
                    )
                )
            )
            # 创建一个名称到对象的映射字典
            self._task_objects[self._cubes[-1].name] = self._cubes[-1]  # self._cubes[-1] 获取刚添加的最后一个立方体

        # 添加Franka机械臂
        try:
            from isaacsim.robot.manipulators.examples.franka import Franka
            self._robot = scene.add(Franka(prim_path="/World/Franka", name="myfranka"))
            self._task_objects[self._robot.name] = self._robot
        except Exception as e:
            print(f"Failed to load Franka robot: {e}")
            print("Continuing without robot for now...")
            self._robot = None
        print("finished adding cubes and robot")
        print(f"创建了 {len(self._cubes)} 个方块")
        print(f"方块名称: {[cube.name for cube in self._cubes]}")
        print(f"方块颜色索引: {self._cube_colors}")
        print(f"目标位置: {self._target_positions}")

    def calculate_metrics(self) -> dict:
        # 计算方块与目标位置距离作为指标
        observations = self.get_observations()
        distances = []
        cube_count = 0
        for key, value in observations.items():
            if key != "target_positions" and key != self._robot.name if self._robot else "franka":
                if "position" in value:
                    cube_count += 1
                    target_idx = (cube_count - 1) % len(self._target_positions)  # 简单分配策略
                    distances.append(np.linalg.norm(value["position"] - self._target_positions[target_idx]))
        return {"avg_distance": np.mean(distances) if distances else 0.0}

    # （2）观察获取（get_observations）
    def get_observations(self) -> dict:

        observations = {}

        # 机械臂观测：关节状态、末端执行器位姿
        if self._robot is not None:
            joints_state = self._robot.get_joints_state()
            joint_currentpos_01 = joints_state.positions
            joint_currentpos_02 = self._robot.get_joint_positions()
            end_effector_position_01, _ = self._robot.end_effector.get_local_pose()
            end_effector_position_02 = self._robot.end_effector.get_world_pose()
            # print("end_effector_position_01", end_effector_position_01)
            # print("end_effector_position_02", end_effector_position_02)
            observations[self._robot.name] = {
                "joint_positions": joint_currentpos_02,
                "end_effector_position": end_effector_position_02,
            }

        # 方块观测：位姿、颜色索引
        for i in range(len(self._cubes)):
            cube_position_local, cube_orientation_local = self._cubes[i].get_local_pose()
            cube_position_world, cube_orientation_world = self._cubes[i].get_world_pose()

            observations[self._cubes[i].name] = {
                "position": cube_position_world,
                "orientation": cube_orientation_world,
                "size": self._cube_scale,
                "color": self._cube_colors[i] if i < len(self._cube_colors) else 0
            }
        self._target_positions /= get_stage_units()
        # 目标位置观测
        observations["target_positions"] = self._target_positions
        return observations

    def reset(self):
        # 重置方块位置 - 使用定点锚点随机摆放（与初始化保持一致）
        if hasattr(self, '_cubes') and self._cubes:
            # 清空之前的位置记录，以便重新生成
            old_positions = self._cube_positions
            old_colors = self._cube_colors
            self._cube_positions = []
            self._cube_colors = []
            
            for i, cube in enumerate(self._cubes):
                # 为每个方块生成新的随机位置和颜色
                new_position, _, new_color = self.add_random_cube()
                
                # 更新方块的物理位置和颜色视觉
                cube.set_world_pose(position=new_position)
                cube.set_default_state(position=new_position)
                # 注意：DynamicCuboid的颜色在创建时固定，无法动态修改
                # 所以我们只更新内部记录，保持逻辑一致
        return True

    def post_reset(self) -> None:
        from isaacsim.robot.manipulators.grippers import ParallelGripper
        if self._robot is not None and isinstance(self._robot.gripper, ParallelGripper):
            # self._robot.gripper.set_gripper_positions(self._robot.gripper.joint_opened_positions)  # 张开夹爪
            print("Gripper opened after reset.")

        return

    def get_params(self) -> dict:
        """Get the parameters of the task."""
        params_representation = dict()
        if self._robot is not None:
            params_representation["robot_name"] = {"value": self._robot.name, "modifiable": False}
        return params_representation

    def is_done(self) -> bool:
        super().is_done()
        # 任务完成条件：所有方块都接近各自目标位置
        observations = self.get_observations()
        for i in range(len(self._cubes)):
            cube_name = self._cubes[i].name
            if cube_name in observations:
                target_idx = i % len(self._target_positions)
                distance = np.linalg.norm(observations[cube_name]["position"] - self._target_positions[target_idx])
                if distance > 0.05:  # 设定一个距离阈值
                    return False
        return True

    # （3）随机生成方块位置、颜色（add_random_cube）
    def add_random_cube(self):
        # 使用定点锚点并加入小扰动进行随机摆放，类似定点程序
        anchors = [
            np.array([0.2, 0.0, 0.03]),
            np.array([0.4, 0.0, 0.03]),
            np.array([0.6, 0.0, 0.03]),
        ]

        max_tries = 50
        for _ in range(max_tries):
            base = anchors[np.random.randint(0, len(anchors))]
            jitter = np.random.uniform(-0.03, 0.03, 3)
            jitter[2] = 0.0  # 不在高度方向扰动太大
            position = base + jitter
            position /= get_stage_units()

            # 检查新位置与所有已存在方块的距离，避免重叠
            if all(np.linalg.norm(position - pos) >= 2 * self._cube_scale[0]
                   for pos in self._cube_positions):
                orientation = None  # 默认朝向
                self._cube_positions.append(position)
                rand_color_idx = np.random.choice(len(self._target_colors))
                cube_color = self._target_colors[rand_color_idx]
                self._cube_colors.append(rand_color_idx)
                return position, orientation, cube_color

        # 退化为均匀采样（如果定点采样多次失败）
        position = np.random.uniform(0.1, 0.5, 3) / get_stage_units()
        orientation = None
        self._cube_positions.append(position)
        rand_color_idx = np.random.choice(len(self._target_colors))
        cube_color = self._target_colors[rand_color_idx]
        self._cube_colors.append(rand_color_idx)
        return position, orientation, cube_color

    def get_cube_names(self):
        """Return the names of all cubes in the task."""
        return [cube.name for cube in self._cubes]


class ArmPickPlaceRLEnv(gym.Env):
    """
    将taskEnv_SceneSetup包装为符合Gymnasium接口的强化学习环境

    观测空间：
        - 机械臂末端执行器位置 (3)
        - 机械臂关节位置 (9, Franka有7个关节+2个夹爪关节)
        - 当前目标方块位置 (3)
        - 当前目标方块颜色索引 (1)
        - 目标放置位置 (3)
        - 夹爪状态 (1)
        总共: 20维观测

    动作空间：
        - 末端执行器在x,y,z方向的增量移动 (3)
        - 夹爪开合动作 (1, -1表示张开, 1表示闭合)
        总共: 4维连续动作
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, headless=False, render_mode=None, max_episode_steps=500, cube_num=6, simulation_app=None):
        """
        初始化RL环境

        参数:
            headless: 是否无头模式运行
            render_mode: 渲染模式
            max_episode_steps: 每个episode最大步数
            cube_num: 场景中方块数量
            simulation_app: 外部传入的SimulationApp实例（可选）
        """
        if not GYMNASIUM_AVAILABLE:
            raise ImportError("gymnasium is required for RL environment. Install with: pip install gymnasium")

        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.cube_num = cube_num

        # 定义观测空间和动作空间
        # 观测: [ee_pos(3), joint_pos(9), cube_pos(3), cube_color(1), target_pos(3), gripper_state(1)]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(20,),
            dtype=np.float32
        )

        # 动作: [delta_x, delta_y, delta_z, gripper_action]
        # delta_x,y,z: 末端执行器位置增量 [-1, 1]
        # gripper_action: -1(张开) 到 1(闭合)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        # 初始化Isaac Sim（如果外部未提供）
        if simulation_app is None:
            from isaacsim import SimulationApp
            self.simulation_app = SimulationApp({"headless": headless})
            self._owns_simulation_app = True
        else:
            self.simulation_app = simulation_app
            self._owns_simulation_app = False

        # 延迟导入（需要在SimulationApp初始化后）
        from isaacsim.core.api import World

        # 创建世界和任务
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()

        self.task = taskEnv_SceneSetup(name="env_armPick", cube_num=self.cube_num)
        self.world.add_task(self.task)

        # 初始化场景
        self.world.reset()

        # 获取机器人
        task_params = self.task.get_params()
        self.robot_name = task_params["robot_name"]["value"]
        self.robot = self.world.scene.get_object(self.robot_name)
        self.articulation_controller = self.robot.get_articulation_controller()

        # 任务相关变量
        self.cube_names = self.task.get_cube_names()
        self.current_cube_idx = 0
        self.has_grasped = False
        self.last_distance_to_cube = None
        self.last_distance_to_target = None

        # 记录初始末端执行器位置
        self.initial_ee_position, _ = self.robot.end_effector.get_world_pose()

    def _get_obs(self):
        """获取当前观测，转换为扁平化的numpy数组"""
        observations = self.task.get_observations()

        # 末端执行器位置
        ee_position, _ = self.robot.end_effector.get_world_pose()

        # 关节位置
        joint_positions = observations[self.robot_name]["joint_positions"]

        # 当前目标方块信息
        if self.current_cube_idx < len(self.cube_names):
            current_cube_name = self.cube_names[self.current_cube_idx]
            cube_position = observations[current_cube_name]["position"]
            cube_color_idx = observations[current_cube_name]["color"]

            # 目标放置位置
            target_position = observations["target_positions"][cube_color_idx]
        else:
            # 所有方块都已处理完成
            cube_position = np.zeros(3)
            cube_color_idx = 0
            target_position = np.zeros(3)

        # 夹爪状态（简化为开合度）
        gripper_state = np.array([1.0 if self.has_grasped else 0.0])

        # 组合观测
        obs = np.concatenate([
            ee_position.astype(np.float32),
            joint_positions.astype(np.float32),
            cube_position.astype(np.float32),
            np.array([cube_color_idx], dtype=np.float32),
            target_position.astype(np.float32),
            gripper_state.astype(np.float32)
        ])

        return obs

    def _compute_reward(self, obs, action):
        """
        简化的奖励函数设计
        
        核心思想：
        1. 使用负指数距离奖励：exp(-k*distance)，提供平滑且持续的梯度
        2. 成功抓取和放置时给予大额奖励
        3. 每个方块的奖励递增，鼓励完成更多方块
        """
        reward = 0.0
        done = False

        if self.current_cube_idx >= len(self.cube_names):
            # 所有方块已完成
            reward = 200.0
            done = True
            return reward, done

        observations = self.task.get_observations()
        current_cube_name = self.cube_names[self.current_cube_idx]
        cube_position = observations[current_cube_name]["position"]
        cube_color_idx = observations[current_cube_name]["color"]
        target_position = observations["target_positions"][cube_color_idx]

        ee_position, _ = self.robot.end_effector.get_world_pose()

        # === 阶段1：未抓取 - 接近方块 ===
        if not self.has_grasped:
            distance_to_cube = np.linalg.norm(ee_position - cube_position)
            
            # 负指数距离奖励：提供持续的平滑梯度
            proximity_reward = 1.0 * np.exp(-5.0 * distance_to_cube)
            reward += proximity_reward

            # 非常接近时的额外奖励
            if distance_to_cube < 0.08:
                reward += 30.0
                
                # 成功抓取！
                if action[3] > 0.5:  # 夹爪闭合
                    self.has_grasped = True
                    # 抓取奖励：基础奖励 + 递增奖励
                    grasp_reward = 100.0 + (self.current_cube_idx * 20.0)
                    reward += grasp_reward
                    print(f"✅ 成功抓取方块 {self.current_cube_idx + 1}/{len(self.cube_names)} "
                          f"(奖励: +{grasp_reward:.1f})")
        
        # === 阶段2：已抓取 - 移动到目标位置 ===
        else:
            distance_to_target = np.linalg.norm(cube_position - target_position)
            
            # 负指数距离奖励（放置阶段奖励更高）
            # exp(-2*d): d=0时奖励=25, d=0.5时奖励≈9.2, d=1时奖励≈3.4
            proximity_reward = 25.0 * np.exp(-2.0 * distance_to_target)
            reward += proximity_reward

            # 非常接近目标位置
            if distance_to_target < 0.08:
                reward += 40.0
                
                # 成功放置！
                if action[3] < -0.5:  # 夹爪张开
                    # 放置奖励：基础奖励 + 大幅递增奖励
                    place_reward = 150.0 + (self.current_cube_idx * 50.0)
                    reward += place_reward
                    
                    print(f"🎉 成功放置方块 {self.current_cube_idx + 1}/{len(self.cube_names)} "
                          f"(奖励: +{place_reward:.1f}, 总进度: {(self.current_cube_idx+1)/len(self.cube_names)*100:.0f}%)")

                    # 移动到下一个方块
                    self.current_cube_idx += 1
                    self.has_grasped = False

                    # 完成所有方块
                    if self.current_cube_idx >= len(self.cube_names):
                        reward += 300.0  # 最终完成的巨大奖励
                        done = True
                        print(f"🏆 任务完成！所有 {len(self.cube_names)} 个方块已分类摆放！")

        # 极小的惩罚（避免无意义的大幅度动作）
        action_penalty = -0.0001 * np.sum(np.abs(action))
        reward += action_penalty

        return reward, done

    def step(self, action):
        """执行一步动作"""
        self.current_step += 1

        # 限制动作幅度
        action = np.clip(action, -1.0, 1.0)

        # 将RL动作转换为机器人控制指令
        # 获取当前末端执行器位置
        ee_position, ee_orientation = self.robot.end_effector.get_world_pose()

        # 计算目标位置（当前位置 + 动作增量）
        # 增大动作尺度以扩大机械臂每步的位移幅度
        action_scale = 0.06  # 控制每步移动的最大距离（从0.02提升到0.06）
        target_ee_position = ee_position + action[:3] * action_scale

        # 控制夹爪
        if action[3] > 0.5:
            self.robot.gripper.close()
        elif action[3] < -0.5:
            self.robot.gripper.open()

        # 使用改进的控制方法
        from isaacsim.core.utils.types import ArticulationAction
        delta_position = action[:3] * action_scale

        # 获取当前关节位置
        current_joint_positions = self.robot.get_joint_positions()

        # 改进的控制：使用更合理的映射，降低增益避免抖动
        target_joint_positions = current_joint_positions.copy()

        # 映射：将动作分量映射到不同关节，使用较小的增益（2.0代替10.0）
        # 降低增益可以减少抖动，同时action_scale=0.06已经提供足够的运动范围
        target_joint_positions[0] += delta_position[1] * 2.0  # y方向 -> joint0
        target_joint_positions[1] += delta_position[2] * 2.0  # z方向 -> joint1
        if len(target_joint_positions) > 3:
            target_joint_positions[3] += delta_position[0] * 2.0  # x方向 -> joint3

        # 使用通用的关节限制以兼容不同Franka实现
        num_joints = len(target_joint_positions)
        joint_limits_lower = np.full(num_joints, -3.1416)
        joint_limits_upper = np.full(num_joints, 3.1416)
        target_joint_positions = np.clip(target_joint_positions, joint_limits_lower, joint_limits_upper)

        # 使用关节位置控制，并设置合理的刚度和阻尼
        robot_action = ArticulationAction(
            joint_positions=target_joint_positions,
            joint_velocities=None  # 让控制器自动计算速度
        )
        self.articulation_controller.apply_action(robot_action)

        # 推进仿真
        self.world.step(render=True)

        # 获取新观测
        obs = self._get_obs()

        # 计算奖励
        reward, done = self._compute_reward(obs, action)

        # 检查是否超过最大步数
        if self.current_step >= self.max_episode_steps:
            done = True
            reward -= 10.0  # 超时惩罚

        # Gymnasium格式: (observation, reward, terminated, truncated, info)
        truncated = self.current_step >= self.max_episode_steps
        info = {
            "current_cube_idx": self.current_cube_idx,
            "has_grasped": self.has_grasped,
            "step": self.current_step
        }

        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)

        # 重置世界和任务（这会触发方块的随机放置）
        self.world.reset()
        self.task.reset()

        # 重置任务相关变量
        self.current_step = 0
        self.current_cube_idx = 0
        self.has_grasped = False
        self.last_distance_to_cube = None
        self.last_distance_to_target = None

        # 更新方块名称列表（因为reset后方块位置已改变）
        self.cube_names = self.task.get_cube_names()

        # 重置机器人状态
        self.robot.gripper.open()

        # 获取初始观测
        obs = self._get_obs()
        info = {}

        return obs, info

    def render(self):
        """渲染环境（Isaac Sim自动渲染）"""
        if self.render_mode == "rgb_array":
            # 可以添加相机截图逻辑
            pass
        # 不需要额外操作，Isaac Sim会自动渲染

    def close(self):
        """关闭环境"""
        # 只有自己创建的SimulationApp才关闭
        if self._owns_simulation_app and hasattr(self, 'simulation_app'):
            self.simulation_app.close()


# 为了向后兼容，保留原来的别名
ArmPickPlaceRLEnv = ArmPickPlaceRLEnv
