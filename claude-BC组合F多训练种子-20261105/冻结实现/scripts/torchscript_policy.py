"""作者导出策略的直接加载与 Go2 部署观测契约核验。"""
from pathlib import Path
import hashlib

import torch


JOINTS = [f"{leg}_{joint}_joint" for joint in ("hip", "thigh", "calf") for leg in ("FL", "FR", "RL", "RR")]
TERMS = ["base_ang_vel", "velocity_commands", "projected_gravity", "joint_pos_history", "joint_vel_history", "actions", "height_map"]
DIMS = [3, 3, 3, 36, 36, 12, 143]
SCALES = [.25, (2., 2., .25), .1, 1., .05, 1., 1.]


def load_exported_policy(path):
    """直接加载原文件，验证归一化和动作输出，禁止补造训练权重。"""
    policy = torch.jit.load(str(path), map_location="cpu").eval()
    state = policy.state_dict()
    if "running_mean" not in state or "running_var" not in state:
        raise ValueError("导出模型缺少已验证的观测归一化参数。")
    dim = state["running_mean"].numel()
    if state["running_var"].shape != (dim,) or (state["running_var"] < 0).any():
        raise ValueError("归一化方差无效。")
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("模型含非有限参数。")
    with torch.inference_mode():
        probe = policy(state["running_mean"].reshape(1, dim))
    if probe.shape != (1, 12) or not torch.isfinite(probe).all():
        raise ValueError("策略输出必须是有限的 12 维关节动作。")
    return policy, {
        "模型路径": str(Path(path).resolve()),
        "模型SHA256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "输入维度": dim, "输出维度": 12,
        "归一化": "原模型内部 (观测-running_mean)/sqrt(running_var+1e-8)，无外部重复归一化",
        "参数形状": {key: list(value.shape) for key, value in state.items()},
    }


def audit_go2_policy_environment(env, policy):
    """逐项检查配置及运行时顺序；保留噪声和动力学供复核。"""
    cfg = env.cfg
    manager = env.observation_manager
    if manager.active_terms["policy"] != TERMS:
        raise ValueError(f"观测顺序不匹配: {manager.active_terms['policy']}")
    dims = [int(torch.tensor(shape).prod()) for shape in manager.group_obs_term_dim["policy"]]
    if dims != DIMS:
        raise ValueError(f"观测块维度不匹配: {dims}")
    blocks = []
    offset = 0
    for name, dim, scale in zip(TERMS, DIMS, SCALES):
        term = getattr(cfg.observations.policy, name)
        torch.testing.assert_close(torch.as_tensor(term.scale), torch.as_tensor(scale))
        blocks.append({"名称": name, "起始索引": offset, "结束索引不含": offset + dim,
                       "缩放": scale, "参数": str(term.params), "噪声": str(term.noise)})
        offset += dim
    for name, mode in (("joint_pos_history", "pos"), ("joint_vel_history", "vel")):
        term = getattr(cfg.observations.policy, name)
        if term.params["names"] != JOINTS or term.params["history_len"] != 3 or term.params["mode"] != mode:
            raise ValueError(f"{name} 的关节顺序或历史配置不符。")
    action = env.action_manager.get_term("joint_pos")
    if action._joint_names != JOINTS or not cfg.actions.joint_pos.use_default_offset:
        raise ValueError("动作关节顺序或默认角度偏移不符。")
    robot = env.scene["robot"]
    ids, _ = robot.find_joints(JOINTS, preserve_order=True)
    default = robot.data.default_joint_pos[0, ids]
    torch.testing.assert_close(default.cpu(), torch.tensor([.1, -.1, .1, -.1, .8, .8, 1., 1., -1.5, -1.5, -1.5, -1.5]))
    if abs(env.step_dt - .02) > 1e-9:
        raise ValueError("部署策略要求 50 赫兹控制频率。")
    ray = cfg.scene.ray_caster
    if ray.ray_alignment != "yaw" or ray.pattern_cfg.ordering != "xy":
        raise ValueError("高程图对齐或展平顺序不符。")
    torch.testing.assert_close(torch.tensor(ray.pattern_cfg.size), torch.tensor([1., .8]))
    if abs(ray.pattern_cfg.resolution - .08) > 1e-9 or tuple(ray.offset.pos) != (.2, 0., .5):
        raise ValueError("高程图分辨率或传感器偏移不符。")
    # 使用独立生成器避免核验改变评估随机数流。与显式网络计算比较，检测重复归一化或权重转换。
    generator = torch.Generator(device="cpu").manual_seed(21349)
    mean, var = policy.running_mean, policy.running_var
    obs = mean + torch.randn((4, 236), generator=generator).to(mean.device) * torch.sqrt(var + 1e-8)
    with torch.inference_mode():
        actual = policy(obs)
        expected = policy.actor((obs - mean) / torch.sqrt(var + 1e-8))
    torch.testing.assert_close(actual, expected)
    actuator = robot.actuators["base_legs"]
    return {
        "观测块": blocks, "历史顺序": "当前帧、前一帧、前两帧；重置时以当前帧填满",
        "关节位置定义": "绝对关节角，未减默认角度", "关节顺序": action._joint_names,
        "默认关节角弧度": default.cpu().tolist(), "动作缩放": cfg.actions.joint_pos.scale,
        "动作公式": "目标关节角=默认关节角+动作缩放*原模型输出",
        "控制周期秒": env.step_dt, "物理时间步秒": cfg.sim.dt,
        "观测噪声启用": cfg.observations.policy.enable_corruption,
        "高程图": ray.to_dict(), "电机配置": actuator.cfg.to_dict(),
        "运行时刚度": actuator.stiffness[0].cpu().tolist(),
        "运行时阻尼": actuator.damping[0].cpu().tolist(),
        "仿真总质量千克": float(robot.root_physx_view.get_masses()[0].sum()),
        "指令范围": cfg.commands.base_velocity.ranges.to_dict(),
        "地形配置": cfg.scene.terrain.terrain_generator.to_dict(),
        "归一化一致性最大绝对误差": float((actual - expected).abs().max()),
    }
