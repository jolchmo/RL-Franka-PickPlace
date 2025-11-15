# franka_pickplace_env_cfg.py
# Isaac Lab环境配置 - Franka抓取任务
# 支持数千个并行环境GPU训练

import math
import torch

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
    
    # -- START: 关键修改在这里 --
    # 从环境的配置(cfg)中获取目标位置列表，并转换为一个与环境数量和设备匹配的张量
    # 注意：这会在每一步都创建张量，对于性能有轻微影响。
    # 更优化的方法是在环境的 __init__ 中创建一次 self.target_pos。但这个方法可以直接解决当前问题。
    target_pos = torch.tensor(env.cfg.target_position, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
    # -- END: 关键修改在这里 --
    # 距离 (1维)
    ee_to_cube = torch.norm(cube_pos - ee_pos, dim=-1, keepdim=True)
    cube_to_target = torch.norm(cube_pos - target_pos, dim=-1, keepdim=True)
    
    # 获取夹持状态 (1维) - 安全获取，处理初始化阶段
    if hasattr(env, 'cube_is_grasped') and env.cube_is_grasped is not None:
        cube_is_grasped = env.cube_is_grasped.unsqueeze(-1)
    else:
        # 初始化阶段默认未抓取
        cube_is_grasped = torch.zeros((env.num_envs, 1), dtype=torch.bool, device=env.device)
    
    # 根据是否抓取来决定关注哪个距离
    distance = torch.where(cube_is_grasped, cube_to_target, ee_to_cube)
    
    # 拼接所有观测 (7+3+3+3+1+1=18维)
    return torch.cat([joint_pos, ee_pos, cube_pos, target_pos, distance, cube_is_grasped.float()], dim=-1)

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


# 定义显式的奖励函数，避兏lamb da闭包问题
def reward_distance_to_cube(env) -> torch.Tensor:
    """接近方块奖励 - 简化版：距离越近奖励越高"""
    if not hasattr(env, 'cube_is_grasped') or not hasattr(env, 'ee_to_cube_dist'):
        return torch.zeros(env.num_envs, device=env.device)
    not_grasped = (~env.cube_is_grasped).float()
    # 使用 (1 - distance) 确保正奖励
    # 距离0.5m: (1-0.5)=0.5, 距离0.1m: (1-0.1)=0.9
    reward = (1.0 - torch.clamp(env.ee_to_cube_dist, 0, 1.0))
    return reward * not_grasped

def reward_distance_to_target(env) -> torch.Tensor:
    """搬运到目标奖励 - 简化版"""
    if not hasattr(env, 'cube_is_grasped') or not hasattr(env, 'cube_to_target_dist'):
        return torch.zeros(env.num_envs, device=env.device)
    grasped = env.cube_is_grasped.float()
    reward = (1.0 - torch.clamp(env.cube_to_target_dist, 0, 1.0))
    return reward * grasped

def reward_reach_cube(env) -> torch.Tensor:
    """接近方块里程碑"""
    if not hasattr(env, 'reach_cube_bonus'):
        return torch.zeros(env.num_envs, device=env.device)
    return env.reach_cube_bonus

def reward_grasp_success(env) -> torch.Tensor:
    """抓取成功里程碑"""
    if not hasattr(env, 'grasp_success_bonus'):
        return torch.zeros(env.num_envs, device=env.device)
    return env.grasp_success_bonus

def reward_place_success(env) -> torch.Tensor:
    """放置成功里程碑"""
    if not hasattr(env, 'place_success_bonus'):
        return torch.zeros(env.num_envs, device=env.device)
    return env.place_success_bonus

def penalty_action(env) -> torch.Tensor:
    """动作大小惩罚"""
    if not hasattr(env, 'actions'):
        return torch.zeros(env.num_envs, device=env.device)
    return -torch.sum(env.actions**2, dim=-1)

@configclass
class RewardsCfg:
    """奖励配置 - 优化版：鼓励积极探索"""
    
    # 距离奖励 - 小权重作为引导
    distance_to_cube = RewardTermCfg(func=reward_distance_to_cube, weight=1.0)
    distance_to_target = RewardTermCfg(func=reward_distance_to_target, weight=2.0)
    
    # 里程碑奖励 - 主要奖励来源
    reach_cube = RewardTermCfg(func=reward_reach_cube, weight=10.0)
    grasp_success = RewardTermCfg(func=reward_grasp_success, weight=50.0)
    place_success = RewardTermCfg(func=reward_place_success, weight=100.0)
    
    # 动作惩罚 - 极小化以鼓励探索
    action_penalty = RewardTermCfg(func=penalty_action, weight=1.0)  # 权重已包含在函数中


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
    
    # 场景设置（num_envs会在train_sb3.py中动态设置）
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
    cube_spawn_height = 0.025  # 方块一半高度
    
    # 任务参数
    grasp_distance_threshold = 0.06  # 触发抓取的距离阈值
    drop_distance_threshold = 0.2    # 判断掉落的距离阈值
    target_success_threshold = 0.05  # 判断放置成功的距离阈值
    
    # 夹爪位置
    gripper_open_pos = 0.04
    gripper_closed_pos = 0.0
    
    # 资产配置
    arm_joint_ids = list(range(7))
    gripper_joint_ids = [7, 8]
    ee_body_name = "panda_hand"
    
    # 动作空间配置（用于环境内部）
    num_actions = 8  # 7个关节 + 1个夹爪
    action_scale_internal = 0.1  # 内部动作缩放
