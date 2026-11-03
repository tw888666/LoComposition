"""把能耗从奖励改为拉格朗日约束的配置改写；在创建环境前调用。"""
import hashlib
import json
import math
from pathlib import Path


def configure_energy_constraint(env_cfg, agent_cfg, reference_path=None, limit_init=None):
    from locomposition.tasks.utils.cat.manager_constraint_cfg import EnergyCostCfg

    if not hasattr(env_cfg, "energy_cost"):
        raise ValueError("当前任务的环境配置没有 energy_cost 字段（目前只支持 Go2 LoComposition 任务）。")
    cfg = agent_cfg.energy_constraint
    reference_info = None
    if reference_path:
        path = Path(reference_path).expanduser().resolve(strict=True)
        reference = json.loads(path.read_text(encoding="utf-8"))
        cfg.power_ref = float(reference["P_ref"])
        reference_info = {"参考尺度文件": str(path), "参考尺度SHA256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if not (math.isfinite(cfg.power_ref) and cfg.power_ref > 0):
        raise ValueError("power_ref 必须是有限正数。")
    if limit_init is not None:
        if not cfg.limit_min <= limit_init <= cfg.limit_max:
            raise ValueError(f"初始功率上限须位于 [{cfg.limit_min}, {cfg.limit_max}] 瓦。")
        cfg.limit_init = float(limit_init)

    removed_weight = None
    if getattr(env_cfg.curriculum, "power", None) is not None:
        removed_weight = env_cfg.curriculum.power.params.get("end_weight")
    env_cfg.rewards.minimize_power = None
    env_cfg.curriculum.power = None
    env_cfg.energy_cost = EnergyCostCfg(tau_s=0.5, power_ref=cfg.power_ref)
    cfg.enabled = True
    if agent_cfg.finetune:
        # 与四组微调 A–E 相同的优化设置，便于同起点、同预算对比
        agent_cfg.learning_rate = 3e-5
        agent_cfg.anneal_lr = False

    manifest = {
        "约束形式": "PPO-Lagrangian（拉格朗日对偶 PPO）：E[平滑机械功率] <= P_lim",
        "版本": "第二版（2026-11-01）：上限只收紧或保持且最多回退一步；lambda 加比例项；跟踪为主判据、地形只作下限；代价以上限为零点",
        "每步代价": ("(平滑功率 - P_lim) / power_ref * step_dt，可正可负；热身期以平滑功率滑动平均为零点"
                     if cfg.cost_centered else "平滑功率 / power_ref * step_dt"),
        "平滑方式": f"指数滑动平均，时间常数 {env_cfg.energy_cost.tau_s} 秒；新回合首步以瞬时功率初始化",
        "优势合成": "(A_r - lambda*A_c) / (1 + lambda)",
        "lambda更新": ("积分项 I <- clip(I + lr*v, 0, lambda_max)，v=(平均平滑功率 - P_lim)/power_ref；"
                       "lambda = clip(I + kp*EMA(v), 0, lambda_max)"),
        "上限调整": ("跟踪 >= perf_ratio*track_ref、地形 >= terrain_floor*terrain_ref 且约束满足时收紧 shrink；"
                     "跟踪 < (perf_ratio-relax_margin)*track_ref 或地形低于下限时回退到上一次收紧前的值（每次收紧最多回退一次），"
                     "并暂停收紧 backoff_cooldown 个判断周期；上限不会超过已到达过的最高值"),
        "已移除奖励项": "minimize_power",
        "原能耗奖励终值权重": removed_weight,
        "初始上限说明": "热身结束时用实测平滑功率滑动平均标定" if cfg.limit_init is None else "命令行指定",
        "学习率": agent_cfg.learning_rate, "学习率退火": agent_cfg.anneal_lr,
        "参数": cfg.to_dict(),
    }
    if reference_info:
        manifest.update(reference_info)
    return manifest
