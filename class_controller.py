# 在 Isaac Sim 中，对控制器的编写，初始化阶段通常包括定义控制器的基本框架、加载必要的模块，以及设置任务所需的初始参数和对象
import typing
import numpy as np
from isaacsim.core.api.controllers import BaseController
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka.controllers.stacking_controller import StackingController
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController
from isaacsim.robot.manipulators.controllers import pick_place_controller
from isaacsim.robot.manipulators.controllers import stacking_controller
#from omni.isaac.franka.controllers import FrankaPickPlaceController



class RLController(BaseController):
    """
    一个接收RL模型并执行其动作的控制器。
    用于在Isaac Sim中部署训练好的强化学习模型。
    """
    def __init__(
        self, 
        name: str, 
        articulation: Articulation,
        vec_normalize=None,
    ) -> None:
        super().__init__(name)
        self._articulation = articulation
        self._model = None
        self._vec_normalize = vec_normalize
    self._action_scale = 0.06  # 动作缩放系数（扩大与环境一致）

    def set_model(self, model, vec_normalize=None):
        """
        用于从外部注入训练好的RL模型
        
        参数:
            model: stable-baselines3训练的模型 (PPO/SAC)
            vec_normalize: VecNormalize对象，用于标准化观测
        """
        self._model = model
        if vec_normalize is not None:
            self._vec_normalize = vec_normalize

    def forward(self, observations: dict) -> ArticulationAction:
        """
        将观察值传递给RL模型，获取动作，并将其转换为机器人指令。
        
        参数:
            observations (dict): 任务环境返回的观测字典
            
        返回:
            ArticulationAction: 机器人关节动作
        """
        if self._model is None:
            # 如果没有模型，则不执行任何操作
            num_dofs = self._articulation.num_dof
            return ArticulationAction(joint_positions=[None] * num_dofs)

        # 1. 将字典观测转换为RL模型需要的扁平化numpy数组
        obs_array = self._dict_to_array(observations)
        
        # 2. 如果使用了VecNormalize，需要标准化观测
        if self._vec_normalize is not None:
            obs_array = self._vec_normalize.normalize_obs(obs_array)
        
        # 3. 使用模型预测动作
        # stable-baselines3的predict方法会返回一个元组(action, state)
        action, _ = self._model.predict(obs_array, deterministic=True)
        # action 是一个numpy数组: [delta_x, delta_y, delta_z, gripper_action]

        # 4. 将RL模型的输出动作转换为机器人可以理解的指令
        
        # 获取当前末端执行器姿态
        current_ee_position, current_ee_orientation = self._articulation.end_effector.get_world_pose()

        # 计算目标位置：当前位置 + 模型输出的位移
        target_ee_position = current_ee_position + action[:3] * self._action_scale
        
        # 夹爪控制
        gripper_action = action[3]
        if gripper_action > 0.5:
            self._articulation.gripper.close()
        elif gripper_action < -0.5:
            self._articulation.gripper.open()

        # 使用改进的控制方法
        delta_position = action[:3] * self._action_scale
        
        # 获取当前关节位置
        current_joint_positions = self._articulation.get_joint_positions()
        
        # 改进的控制：使用更合理的映射，降低增益避免抖动
        target_joint_positions = current_joint_positions.copy()
        
        # 使用较小的增益（2.0代替10.0）来减少抖动
        target_joint_positions[0] += delta_position[1] * 2.0  # y方向
        target_joint_positions[1] += delta_position[2] * 2.0  # z方向
        if len(target_joint_positions) > 3:
            target_joint_positions[3] += delta_position[0] * 2.0  # x方向

        # 使用通用的关节限制以避免数组长度不匹配问题
        num_joints = len(target_joint_positions)
        joint_limits_lower = np.full(num_joints, -3.1416)
        joint_limits_upper = np.full(num_joints, 3.1416)
        target_joint_positions = np.clip(target_joint_positions, joint_limits_lower, joint_limits_upper)
        
        return ArticulationAction(joint_positions=target_joint_positions)
    # class_controller.py -> RLController._dict_to_array()

    def _dict_to_array(self, observations: dict) -> np.ndarray:
        """
        将观测字典转换为扁平化的numpy数组
        *** 必须与 rl_environment.py 中的 _get_obs() 完全一致 ***
        """
        # 提取机器人观测
        robot_obs = observations[self._articulation.name]
        ee_position = robot_obs["end_effector_position"]
        joint_positions = robot_obs["joint_positions"]
        
        # 提取当前目标方块和目标位置信息
        cube_position = observations.get("current_cube_position", np.zeros(3))
        target_position = observations.get("current_target_position", np.zeros(3))
        has_grasped = observations.get("gripper_state", 0.0) # 0.0 for open, 1.0 for grasped

        # 关键：完全复制训练环境中的相对向量计算逻辑
        if not has_grasped:
            relative_vec = cube_position - ee_position
        else:
            relative_vec = target_position - cube_position
        
        distance = np.linalg.norm(relative_vec)
        
        gripper_state_array = np.array([has_grasped])

        # 组合观测，顺序和维度必须与训练时完全一致
        obs_array = np.concatenate([
            ee_position.astype(np.float32),              # 3
            joint_positions.astype(np.float32),            # 9 (7 arm + 2 gripper)
            relative_vec.astype(np.float32),         # 3
            np.array([distance], dtype=np.float32), # 1
            cube_position.astype(np.float32),             # 3
            target_position.astype(np.float32),           # 3
            gripper_state_array.astype(np.float32)       # 1
        ]) # Total: 3+9+3+1+3+3+1 = 23
        
        return obs_array

    
    def reset(self):
        """重置控制器状态"""
        super().reset()
        # RL控制器通常不需要特殊的重置逻辑
        pass

