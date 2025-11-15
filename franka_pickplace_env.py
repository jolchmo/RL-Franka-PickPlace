# franka_pickplace_env.py
# Isaac Lab环境实现 - Franka抓取任务

import torch
import math
try:
    from isaaclab.envs import ManagerBasedRLEnv
except ImportError:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
from franka_pickplace_env_cfg import FrankaPickPlaceEnvCfg


class FrankaPickPlaceEnv(ManagerBasedRLEnv):
    """
    Franka机器人抓取任务环境
    支持数千个并行环境GPU训练
    """
    
    cfg: FrankaPickPlaceEnvCfg
    
    def __init__(self, cfg: FrankaPickPlaceEnvCfg, render_mode: str = None, **kwargs):
        # 任务状态必须在super之前初始化（observation函数会用到）
        # 使用临时值，因为device还不可用
        self.cube_grasped = None
        self.grasp_triggered = None
        self.target_pos = None
        self.ee_to_cube_distance = None
        self.cube_to_target_distance = None
        self.reach_cube_bonus = None
        self.grasp_success_bonus = None
        self.place_success_bonus = None
        self.task_success = None
        
        # 存储配置中的目标位置
        self._cfg_target_pos = cfg.target_position
        
        super().__init__(cfg, render_mode, **kwargs)
        
        # 获取场景中的资产引用（在super().__init__之后，scene已经设置好了）
        self.robot = self.scene["robot"]
        self.cube = self.scene["cube"]
        
        # 现在device可用了，正确初始化所有tensor
        self.cube_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.grasp_triggered = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 目标位置
        self.target_pos = torch.tensor(
            self._cfg_target_pos, 
            dtype=torch.float32, 
            device=self.device
        ).repeat(self.num_envs, 1)
        
        # 距离缓存
        self.ee_to_cube_distance = torch.zeros(self.num_envs, device=self.device)
        self.cube_to_target_distance = torch.zeros(self.num_envs, device=self.device)
        
        # 奖励标志
        self.reach_cube_bonus = torch.zeros(self.num_envs, device=self.device)
        self.grasp_success_bonus = torch.zeros(self.num_envs, device=self.device)
        self.place_success_bonus = torch.zeros(self.num_envs, device=self.device)
        self.task_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
    def _setup_scene(self):
        """设置场景"""
        # 父类会自动加载scene配置中的所有资产
        super()._setup_scene()
        
    def _pre_physics_step(self, actions: torch.Tensor):
        """物理步进前的动作处理"""
        # 缩放动作
        scaled_actions = actions * self.cfg.action_scale
        
        # 获取当前关节位置
        current_joint_pos = self.robot.data.joint_pos[:, :7]
        
        # 计算目标关节位置
        target_joint_pos = current_joint_pos + scaled_actions
        
        # 应用到机器人
        self.robot.set_joint_position_target(target_joint_pos, joint_ids=list(range(7)))
        
        # 处理夹爪
        gripper_command = torch.where(
            self.cube_grasped.unsqueeze(-1),
            torch.zeros((self.num_envs, 2), device=self.device),  # 闭合
            torch.ones((self.num_envs, 2), device=self.device) * 0.04,  # 打开
        )
        self.robot.set_joint_position_target(gripper_command, joint_ids=[7, 8])
        
    def _apply_action(self):
        """应用动作（由父类调用）"""
        pass  # 动作在_pre_physics_step中已处理
        
    def _get_observations(self) -> dict:
        """获取观测"""
        # 末端执行器位置
        ee_pos = self.robot.data.body_pos_w[:, self.robot.ee_body_idx, :]
        
        # 方块位置
        cube_pos = self.cube.data.root_pos_w
        
        # 关节位置
        joint_pos = self.robot.data.joint_pos[:, :7]
        
        # 计算距离
        self.ee_to_cube_distance = torch.norm(cube_pos - ee_pos, dim=-1)
        self.cube_to_target_distance = torch.norm(cube_pos - self.target_pos, dim=-1)
        
        # 选择当前目标
        current_target = torch.where(
            self.cube_grasped.unsqueeze(-1).expand(-1, 3),
            self.target_pos,
            cube_pos,
        )
        
        distance = torch.where(
            self.cube_grasped,
            self.cube_to_target_distance,
            self.ee_to_cube_distance,
        ).unsqueeze(-1)
        
        # 夹爪状态
        gripper_state = self.cube_grasped.float().unsqueeze(-1)
        
        # 拼接观测
        obs = torch.cat([
            joint_pos,
            ee_pos,
            cube_pos,
            current_target,
            distance,
            gripper_state,
        ], dim=-1)
        
        return {"policy": obs}
        
    def _get_rewards(self) -> torch.Tensor:
        """计算奖励"""
        # 重置奖励标志
        self.reach_cube_bonus.zero_()
        self.grasp_success_bonus.zero_()
        self.place_success_bonus.zero_()
        
        # 获取位置
        ee_pos = self.robot.data.body_pos_w[:, self.robot.ee_body_idx, :]
        cube_pos = self.cube.data.root_pos_w
        
        # 未抓取阶段
        not_grasped_mask = ~self.cube_grasped
        if not_grasped_mask.any():
            # 接近奖励（距离<6cm）
            close_mask = (self.ee_to_cube_distance < 0.06) & not_grasped_mask
            self.reach_cube_bonus[close_mask] = 1.0
            
            # 抓取判定（距离<4.5cm）
            very_close_mask = (self.ee_to_cube_distance < 0.045) & not_grasped_mask & ~self.grasp_triggered
            if very_close_mask.any():
                self.cube_grasped[very_close_mask] = True
                self.grasp_triggered[very_close_mask] = True
                
                # 检查抬起（高度>5cm）
                lifted_mask = (cube_pos[:, 2] > 0.05) & very_close_mask
                self.grasp_success_bonus[lifted_mask] = 1.0
        
        # 抓取后阶段
        grasped_mask = self.cube_grasped
        if grasped_mask.any():
            # 放置判定（距离<8cm且高度<6cm）
            place_mask = (self.cube_to_target_distance < 0.08) & (cube_pos[:, 2] < 0.06) & grasped_mask
            if place_mask.any():
                self.place_success_bonus[place_mask] = 1.0
                self.task_success[place_mask] = True
        
        # 距离奖励会由RewardsCfg自动计算
        return torch.zeros(self.num_envs, device=self.device)  # 基础奖励，其他由manager处理
        
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """判断终止"""
        # 会由TerminationsCfg自动处理
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device), torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
    def _reset_idx(self, env_ids: torch.Tensor):
        """重置指定环境"""
        super()._reset_idx(env_ids)
        
        num_resets = len(env_ids)
        
        # 重置状态
        self.cube_grasped[env_ids] = False
        self.grasp_triggered[env_ids] = False
        self.task_success[env_ids] = False
        
        # 随机化方块位置（相对于每个环境的原点）
        # 获取机器人基座的位置（每个环境的原点）
        robot_base_pos = self.robot.data.root_pos_w[env_ids].clone()
        
        # 在机器人前方生成方块（局部坐标）
        local_cube_pos = torch.zeros((num_resets, 3), device=self.device)
        local_cube_pos[:, 0] = torch.rand(num_resets, device=self.device) * \
            (self.cfg.cube_spawn_x_range[1] - self.cfg.cube_spawn_x_range[0]) + \
            self.cfg.cube_spawn_x_range[0]
        local_cube_pos[:, 1] = torch.rand(num_resets, device=self.device) * \
            (self.cfg.cube_spawn_y_range[1] - self.cfg.cube_spawn_y_range[0]) + \
            self.cfg.cube_spawn_y_range[0]
        local_cube_pos[:, 2] = 0.025  # 方块高度一半（在桌面上）
        
        # 转换为世界坐标（相对于机器人基座）
        cube_pos = robot_base_pos + local_cube_pos
        
        # 设置方块位置和姿态
        cube_pose = torch.cat([
            cube_pos, 
            torch.tensor([[1, 0, 0, 0]], device=self.device).repeat(num_resets, 1)  # 单位四元数
        ], dim=-1)
        
        self.cube.write_root_pose_to_sim(cube_pose, env_ids=env_ids)
        
        # 重置机器人到默认姿态
        default_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        default_joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
