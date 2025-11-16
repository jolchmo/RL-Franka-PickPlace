# franka_pickplace_env_v2.py
# 经过修正和重构的 Isaac Lab 环境实现 - Franka 抓取任务

import torch
from typing import Tuple


from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import RigidObject


from franka_pickplace_env_cfg import FrankaPickPlaceEnvCfg

class FrankaPickPlaceEnv(ManagerBasedRLEnv):
    """
    Franka机器人抓取并放置任务的环境（修正版）。

    此版本修复了原始版本中的多个关键错误，并遵循 Isaac Lab 的设计范式：
    1. 奖励和终止完全由配置文件的 Manager 管理。
    2. 状态更新在物理步进之后进行 (`_update_buffers`)。
    3. 夹爪由智能体动作明确控制（8-DoF 动作空间）。
    4. 使用运动学方式将抓取的物体附加到夹爪，以实现稳定抓取。
    """

    cfg: FrankaPickPlaceEnvCfg

    def __init__(self, cfg: FrankaPickPlaceEnvCfg, render_mode: str = None, **kwargs):
        """
        在父类初始化之后，初始化所有特定于任务的缓冲区和状态。
        """
        # 调用父类构造函数，它会设置场景、设备等
        super().__init__(cfg, render_mode, **kwargs)

        # -- 获取资产引用 --
        self.robot = self.scene["robot"]
        self.cube = self.scene["cube"]
        
        # 获取末端执行器的body索引
        self.ee_body_idx = self.robot.find_bodies(self.cfg.ee_body_name)[0]

        # -- 初始化任务状态缓冲区 --
        # 动作缓存（8-DoF: 7 for arm, 1 for gripper）
        self.actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        
        # 核心任务状态
        self.cube_is_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.task_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 新增：用于里程碑奖励的事件追踪
        self.just_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.just_placed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 新增：尝试抓取状态（用于奖励塑造）
        self.is_attempting_grasp = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 目标位置
        self.target_pos = torch.tensor(self.cfg.target_position, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)

        # 缓存的物理状态，避免在单步内重复访问sim数据
        self.ee_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.cube_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_to_cube_dist = torch.zeros(self.num_envs, device=self.device)
        self.cube_to_target_dist = torch.zeros(self.num_envs, device=self.device)
        
        # 用于运动学抓取的相对变换矩阵
        self._ee_T_cube = torch.zeros(self.num_envs, 4, 4, device=self.device)

    def _setup_scene(self) -> None:
        """设置场景。父类会自动加载所有资产。"""
        super()._setup_scene()
        # 也可以在这里添加额外的场景设置代码

    def _update_buffers(self, dt: float) -> None:
        """在每个物理步进后更新缓冲区和任务状态。"""
        # 调用父类的方法（如果存在）
        # super()._update_buffers(dt) # 如果父类有此方法

        # 0. 缓存上一步的状态（在所有更新之前）
        prev_cube_is_grasped = self.cube_is_grasped.clone()
        prev_task_success = self.task_success.clone()

        # 1. 更新缓存的物理状态
        self.ee_pos_w = self.robot.data.body_pos_w[:, self.ee_body_idx]
        self.cube_pos_w = self.cube.data.root_pos_w
        self.ee_to_cube_dist = torch.norm(self.ee_pos_w - self.cube_pos_w, dim=-1)
        self.cube_to_target_dist = torch.norm(self.target_pos - self.cube_pos_w, dim=-1)

        # 2. 更新抓取状态 (grasp state)
        # 提取夹爪动作（第8个动作，索引7）
        gripper_action = self.actions[:, 7]
        
        # 新增：更新“尝试抓取”状态（用于奖励塑造）
        # 条件：靠近方块 + 夹爪命令为闭合 + 当前还未抓取
        self.is_attempting_grasp = (
            (self.ee_to_cube_dist < self.cfg.grasp_distance_threshold) &
            (gripper_action > 0) &
            ~self.cube_is_grasped
        )
        
        # 条件：何时触发抓取
        # - 夹爪正在尝试闭合 (gripper_action > 0)
        # - 末端执行器足够接近方块
        # - 当前还未抓取方块
        attempt_grasp_mask = (gripper_action > 0) & (self.ee_to_cube_dist < self.cfg.grasp_distance_threshold) & ~self.cube_is_grasped
        
        # 条件：何时触发释放
        # - 夹爪正在尝试打开 (gripper_action <= 0)
        # - 当前正抓着方塊
        attempt_release_mask = (gripper_action <= 0) & self.cube_is_grasped

        # 物理检查：如果已经“抓取”但方块离得太远（例如，由于碰撞而掉落），则强制释放
        dropped_mask = (self.ee_to_cube_dist > self.cfg.drop_distance_threshold) & self.cube_is_grasped
        
        # 更新抓取状态
        # 设置抓取
        if torch.any(attempt_grasp_mask):
            self.cube_is_grasped[attempt_grasp_mask] = True
            self._compute_grasp_transform(attempt_grasp_mask)
            # 禁用方块的物理特性，以进行运动学附加
            self.cube.disable_physics(env_ids=torch.where(attempt_grasp_mask)[0])

        # 设置释放
        release_mask = attempt_release_mask | dropped_mask
        if torch.any(release_mask):
            self.cube_is_grasped[release_mask] = False
            # 重新启用方块的物理特性
            self.cube.enable_physics(env_ids=torch.where(release_mask)[0])

        # 3. 更新任务成功状态 (用于终止条件)
        # 成功条件：抓着方块并且非常接近目标位置
        self.task_success = (self.cube_to_target_dist < self.cfg.target_success_threshold) & self.cube_is_grasped

        # 4. 计算里程碑事件（用于奖励）
        # just_grasped: 从未抓取变为已抓取
        self.just_grasped = ~prev_cube_is_grasped & self.cube_is_grasped
        # just_placed: 从未成功变为成功
        self.just_placed = ~prev_task_success & self.task_success
        
        # 调试输出：打印关键事件（每500步打印一次）
        if hasattr(self, '_debug_step_count') and self._debug_step_count % 500 == 0:
            print(f"[Debug] Step {self._debug_step_count} Stats:")
            print(f"  - Attempting grasp: {self.is_attempting_grasp.sum().item()}/{self.num_envs} envs")
            print(f"  - Grasped cubes: {self.cube_is_grasped.sum().item()}/{self.num_envs} envs")
            print(f"  - Just grasped: {self.just_grasped.sum().item()} envs this step")
            print(f"  - Avg distance to cube: {self.ee_to_cube_dist.mean().item():.4f}m")

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """在物理步进前应用动作并处理运动学。
        
        注意：由于ActionManager配置了9个关节（7臂+2指），它会自动应用前7个动作到手臂关节。
        这里我们额外提取夹爪控制信号（action[7]或action[8]）来判断抓取意图。
        """
        # 缓存动作（注意：actions可能是9维的，由ActionManager处理）
        # 我们只关心前8维：7个手臂增量 + 1个夹爪信号
        if actions.shape[1] >= 8:
            self.actions = actions[:, :8].clone()
        else:
            # 兼容性：如果只有7维，填充0
            self.actions = torch.zeros(self.num_envs, 8, device=self.device)
            self.actions[:, :actions.shape[1]] = actions
        
        # 调试输出：打印第一个环境的夹爪动作值（每100步打印一次）
        if not hasattr(self, '_debug_step_count'):
            self._debug_step_count = 0
        self._debug_step_count += 1
        
        if self._debug_step_count % 100 == 0 and self.num_envs > 0:
            gripper_val = self.actions[0, 7].item()
            print(f"[Debug] Step {self._debug_step_count}: Gripper Action (Env 0) = {gripper_val:.4f}")

        # --- 1. 应用机器人动作（ActionManager已处理手臂关节） ---
        # 我们只需要手动处理夹爪的开合逻辑
        
        # 提取夹爪控制信号（第8个动作，索引7）
        gripper_action = self.actions[:, 7]

        # 应用夹爪动作 (二元开合)
        # gripper_action > 0 表示想要闭合（抓取）
        # gripper_action <= 0 表示想要打开（释放）
        gripper_target_pos = torch.where(
            gripper_action.unsqueeze(-1) > 0, 
            torch.full((self.num_envs, 2), self.cfg.gripper_closed_pos, device=self.device),  # 闭合
            torch.full((self.num_envs, 2), self.cfg.gripper_open_pos, device=self.device)     # 打开
        )
        self.robot.set_joint_position_target(gripper_target_pos, joint_ids=self.cfg.gripper_joint_ids)

        # --- 2. 处理运动学附加 ---
        # 如果方块被抓取，则手动更新其位置以跟随末端执行器
        grasped_env_ids = torch.where(self.cube_is_grasped)[0]
        if len(grasped_env_ids) > 0:
            # 获取当前末端执行器的变换矩阵
            ee_T_w = self.robot.data.body_state_w[:, self.ee_body_idx, :7]
            ee_tf_w = RigidObject.poses_to_transform_matrices(ee_T_w[grasped_env_ids])

            # 计算方块新的世界变换矩阵: T_world_cube = T_world_ee * T_ee_cube
            cube_tf_w = ee_tf_w @ self._ee_T_cube[grasped_env_ids]
            
            # 转换为位姿 [pos, quat]
            new_cube_pose = RigidObject.transform_matrices_to_poses(cube_tf_w)
            
            # 直接写入仿真器，覆盖物理计算
            self.cube.write_root_pose_to_sim(new_cube_pose, env_ids=grasped_env_ids)

    def _apply_action(self) -> None:
        """空函数，因为所有动作逻辑都在 _pre_physics_step 中处理。"""
        pass

    def _get_observations(self) -> dict:
        """收集并拼接所有观测数据。"""
        # 机器人关节位置
        joint_pos = self.robot.data.joint_pos[:, self.cfg.arm_joint_ids]
        
        # 根据是否抓取来选择当前的目标
        # 如果未抓取，目标是方块；如果已抓取，目标是最终放置点
        current_target_pos = torch.where(
            self.cube_is_grasped.unsqueeze(-1), 
            self.target_pos, 
            self.cube_pos_w
        )
        
        # 拼接所有观测信息
        obs = torch.cat(
            (
                joint_pos,                  # 7 DoF
                self.ee_pos_w,              # 3 DoF
                self.cube_pos_w,             # 3 DoF
                current_target_pos,         # 3 DoF
                self.ee_to_cube_dist.unsqueeze(-1), # 1 DoF
                self.cube_to_target_dist.unsqueeze(-1), # 1 DoF
                self.cube_is_grasped.float().unsqueeze(-1), # 1 DoF
            ),
            dim=-1,
        )

        return {"policy": obs}

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """重置指定环境的状态。"""
        # 调用父类的重置，它会处理场景中资产的默认重置
        super()._reset_idx(env_ids)

        num_resets = len(env_ids)
        
        # --- 重置任务相关的状态 ---
        self.cube_is_grasped[env_ids] = False
        self.task_success[env_ids] = False
        
        # --- 随机化方块位置 ---
        # 获取对应环境的原点
        env_origins = self.scene.env_origins[env_ids]
        
        # 在环境的局部坐标系中随机生成位置
        local_cube_pos = torch.zeros((num_resets, 3), device=self.device)
        local_cube_pos[:, 0] = torch.rand(num_resets, device=self.device) * \
            (self.cfg.cube_spawn_x_range[1] - self.cfg.cube_spawn_x_range[0]) + self.cfg.cube_spawn_x_range[0]
        local_cube_pos[:, 1] = torch.rand(num_resets, device=self.device) * \
            (self.cfg.cube_spawn_y_range[1] - self.cfg.cube_spawn_y_range[0]) + self.cfg.cube_spawn_y_range[0]
        local_cube_pos[:, 2] = self.cfg.cube_spawn_height

        # 转换为世界坐标并设置姿态
        cube_pos_w = env_origins + local_cube_pos
        default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device) # w, x, y, z
        cube_pose = torch.cat([cube_pos_w, default_quat.expand(num_resets, 4)], dim=-1)
        
        self.cube.write_root_pose_to_sim(cube_pose, env_ids=env_ids)
        self.cube.write_root_velocity_to_sim(torch.zeros_like(cube_pose[:, :6]), env_ids=env_ids)
        
        # 🔧 修复: Reset后立即更新缓冲区,确保第一步使用正确的状态
        # 只更新被reset的环境的状态
        self.ee_pos_w[env_ids] = self.robot.data.body_pos_w[env_ids, self.ee_body_idx]
        self.cube_pos_w[env_ids] = self.cube.data.root_pos_w[env_ids]
        self.ee_to_cube_dist[env_ids] = torch.norm(self.ee_pos_w[env_ids] - self.cube_pos_w[env_ids], dim=-1)
        self.cube_to_target_dist[env_ids] = torch.norm(self.target_pos[env_ids] - self.cube_pos_w[env_ids], dim=-1)

    def _compute_grasp_transform(self, env_ids: torch.Tensor):
        """计算并存储从末端执行器到方块的相对变换矩阵。"""
        # 获取 EE 和 Cube 的变换矩阵
        ee_T_w = self.robot.data.body_state_w[:, self.ee_body_idx, :7]
        ee_tf_w = RigidObject.poses_to_transform_matrices(ee_T_w[env_ids])
        
        cube_T_w = self.cube.data.root_state_w[env_ids, :7]
        cube_tf_w = RigidObject.poses_to_transform_matrices(cube_T_w)

        # 计算相对变换: T_ee_cube = T_ee_world * T_world_cube
        ee_tf_w_inv = torch.inverse(ee_tf_w)
        self._ee_T_cube[env_ids] = ee_tf_w_inv @ cube_tf_w

