"""同一起点、同预算的能耗微调配置；E用于隔离C的力矩平方项；F为B的权重加C的10%力矩平方混合。"""
import hashlib
import json
import math
from pathlib import Path


BRANCHES = {"A": (1.0, 0.0), "B": (1.5, 0.0), "C": (1.0, 0.1), "D": (1.0, 0.3),
            "E": (0.9, 0.0), "F": (1.5, 0.1)}


def configure_energy_finetune(env_cfg, agent_cfg, branch, reference_path):
    from locomposition.tasks.utils.mdp.rewards import joint_energy_proxy

    if not reference_path:
        raise ValueError("必须指定四组共用的 --energy_reference。")
    path = Path(reference_path).expanduser().resolve(strict=True)
    reference = json.loads(path.read_text())
    power_ref = float(reference["P_ref"])
    torque_ref = float(reference["Q_ref"])
    if not all(math.isfinite(x) and x > 0 for x in (power_ref, torque_ref)):
        raise ValueError("固定参考尺度必须是有限正数。")
    checkpoint = Path(agent_cfg.load_checkpoint).expanduser().resolve(strict=True)
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_hash != reference["checkpoint_sha256"]:
        raise ValueError("微调起点与计算固定参考尺度的原模型不一致。")
    original_weight = float(env_cfg.curriculum.power.params["end_weight"])
    if not math.isclose(original_weight, 0.008, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"基线能耗权重发生变化，需重新核对：{original_weight}")
    multiplier, alpha = BRANCHES[branch]
    weight = original_weight * multiplier
    term = env_cfg.rewards.minimize_power
    term.func = joint_energy_proxy
    term.weight = weight
    term.params = {"scaling_factor": 1.0, "mix_alpha": alpha, "torque_scale": power_ref / torque_ref}
    env_cfg.curriculum.power.params["start_weight"] = weight
    env_cfg.curriculum.power.params["end_weight"] = weight
    agent_cfg.learning_rate = 3e-5
    agent_cfg.anneal_lr = False
    return {
        "实验组": branch, "来源检查点": str(checkpoint), "来源SHA256": checkpoint_hash,
        "追加迭代": agent_cfg.num_iterations, "随机种子": agent_cfg.seed,
        "环境数量": env_cfg.scene.num_envs, "每轮控制步": agent_cfg.num_steps,
        "学习率": agent_cfg.learning_rate, "学习率退火": agent_cfg.anneal_lr,
        "原能耗权重": original_weight, "能耗权重": weight, "损耗占比": alpha,
        "机械功率参考瓦": power_ref, "力矩平方参考": torque_ref, "力矩尺度": power_ref / torque_ref,
        "参考尺度文件": str(path), "参考尺度SHA256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "目标": "-w*((1-alpha)*sum(abs(tau*dq))+alpha*s*sum(tau**2))",
        "优化器": "重新建立Adam，不恢复原优化器状态", "约束课程": "从首次重置前固定为成熟阶段1.0",
        "环境初始化": "同种子重新初始化物理状态、地形课程与约束归一化，不恢复原轨迹",
        "解释边界": "力矩平方是损耗代理；统计尺度s不是真实电机参数。",
    }
