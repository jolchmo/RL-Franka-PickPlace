# franka_pickplace_env_cfg.py
# Isaac Lab环境配置 - Franka抓取任务
# 支持数千个并行环境GPU训练

import math
import torch
try:
    from isaaclab.utils import configclass
    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.managers import SceneEntityCfg, EventTermCfg, RewardTermCfg, TerminationTermCfg, ObservationTermCfg, ActionTermCfg, ObservationGroupCfg
    from isaaclab.envs.mdp.actions import JointPositionActionCfg
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
    from isaaclab.sensors import ContactSensorCfg
    import isaaclab.sim as sim_utils
    from isaaclab_assets import FRANKA_PANDA_CFG
except ImportError:
    from omni.isaac.lab.utils import configclass
    from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
    from omni.isaac.lab.scene import InteractiveSceneCfg
    from omni.isaac.lab.sim import SimulationCfg
    from omni.isaac.lab.managers import SceneEntityCfg, EventTermCfg, RewardTermCfg, TerminationTermCfg, ObservationTermCfg, ObservationGroupCfg
    from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
    from omni.isaac.lab.sensors import ContactSensorCfg
    import omni.isaac.lab.sim as sim_utils
    from omni.isaac.lab_assets import FRANKA_PANDA_CFG


##
# Scene配置
##

@configclass
class FrankaPickPlaceSceneCfg(InteractiveSceneCfg):
    """场景配置：包含机器人、方块、地面等"""
    
    # 地面
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    
    # Franka机器人
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # 方块
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),  # 5cm立方体
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.2, 0.2),  # 红色
                metallic=0.2,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.025)),
    )
    



##
# MDP设置
##

def get_observations(env):
    """获取所有观测并拼接"""
    # 关节位置 (7维)
    joint_pos = env.scene["robot"].data.joint_pos[:, :7]
    # 末端执行器位置 (3维)
    ee_pos = env.scene["robot"].data.body_pos_w[:, -1, :]
    # 方块位置 (3维)
    cube_pos = env.scene["cube"].data.root_pos_w
    # 目标位置 (3维) - 在初始化前使用临时值
    target_pos = env.target_pos
    if target_pos is None:
        # 初始化阶段使用默认目标位置
        target_pos = torch.tensor([[0.5, 0.2, 0.0]], dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
    # 距离 (1维)
    ee_to_cube = torch.norm(cube_pos - ee_pos, dim=-1, keepdim=True)
    cube_to_target = torch.norm(cube_pos - target_pos, dim=-1, keepdim=True)
    # 安全获取cube_grasped属性（处理None的情况）
    cube_grasped = getattr(env, 'cube_grasped', None)
    if cube_grasped is None:
        cube_grasped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    distance = torch.where(
        cube_grasped.unsqueeze(-1),
        cube_to_target,
        ee_to_cube
    )
    # 夹爪状态 (1维)
    gripper = cube_grasped.float().unsqueeze(-1)
    
    # 拼接所有观测 (7+3+3+3+1+1=18维)
    return torch.cat([joint_pos, ee_pos, cube_pos, target_pos, distance, gripper], dim=-1)

@configclass 
class ObservationsCfg:
    """观测配置"""
    
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Policy网络的观测"""
        concatenate_terms = True
        enable_corruption = False
        
        @configclass
        class AllObsCfg(ObservationTermCfg):
            func = get_observations
        
        all_obs: AllObsCfg = AllObsCfg()
    
    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    """动作配置：7个关节的位置增量"""
    joint_pos: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.5,
        use_default_offset=True
    )


@configclass
class RewardsCfg:
    """奖励配置"""
    
    # 距离奖励
    distance_to_cube = RewardTermCfg(func=lambda env: -env.ee_to_cube_distance, weight=1.0)
    distance_to_target = RewardTermCfg(func=lambda env: -env.cube_to_target_distance, weight=2.0)
    
    # 里程碑奖励
    reach_cube = RewardTermCfg(func=lambda env: env.reach_cube_bonus, weight=20.0)
    grasp_success = RewardTermCfg(func=lambda env: env.grasp_success_bonus, weight=100.0)
    place_success = RewardTermCfg(func=lambda env: env.place_success_bonus, weight=150.0)
    
    # 惩罚
    action_penalty = RewardTermCfg(func=lambda env: -0.01, weight=1.0)


@configclass
class TerminationsCfg:
    """终止条件配置"""
    
    # 超时
    time_out = TerminationTermCfg(func=lambda env: env.episode_length_buf >= env.max_episode_length, time_out=True)
    
    # 成功完成
    success = TerminationTermCfg(func=lambda env: env.task_success)


##
# 环境配置
##

@configclass
class FrankaPickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Franka抓取任务环境配置"""
    
    # 场景设置
    scene: FrankaPickPlaceSceneCfg = FrankaPickPlaceSceneCfg(num_envs=4096, env_spacing=2.0)
    
    # 基础设置
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    
    # 仿真设置
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 60.0,  # 60Hz
        render_interval=2,  # 每2个物理步渲染一次
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    
    # Episode设置
    episode_length_s = 10.0  # 10秒
    decimation = 2  # 控制频率 = 60Hz / 2 = 30Hz
    
    # 动作缩放
    action_scale = 0.5
    
    # 目标位置（米）
    target_position = [0.5, 0.2, 0.0]
    
    # 方块初始位置范围
    cube_spawn_x_range = [0.3, 0.5]
    cube_spawn_y_range = [-0.2, 0.2]
