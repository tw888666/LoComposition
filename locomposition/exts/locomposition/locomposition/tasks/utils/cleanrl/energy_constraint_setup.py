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
    unit = "焦耳/米" if cfg.cost_metric == "per_distance" else "瓦"
    if limit_init is not None:
        if not cfg.limit_min <= limit_init <= cfg.limit_max:
            raise ValueError(f"初始上限须位于 [{cfg.limit_min}, {cfg.limit_max}] {unit}。")
        cfg.limit_init = float(limit_init)
    if not (cfg.speed_offset > 0 and cfg.yaw_rate_weight >= 0 and cfg.lateral_penalty >= 0
            and cfg.yaw_error_penalty >= 0 and cfg.perp_cmd_threshold >= 0):
        raise ValueError("speed_offset 必须为正，其余速度权重与阈值不得为负。")
    if cfg.metric_ref is not None and not (math.isfinite(cfg.metric_ref) and cfg.metric_ref > 0):
        raise ValueError("metric_ref 必须为有限正数或 None。")

    removed_weight = None
    if getattr(env_cfg.curriculum, "power", None) is not None:
        removed_weight = env_cfg.curriculum.power.params.get("end_weight")
    env_cfg.rewards.minimize_power = None
    env_cfg.curriculum.power = None
    env_cfg.energy_cost = EnergyCostCfg(tau_s=0.5, power_ref=cfg.power_ref,
                                        speed_offset=cfg.speed_offset, yaw_rate_weight=cfg.yaw_rate_weight,
                                        lateral_penalty=cfg.lateral_penalty, yaw_error_penalty=cfg.yaw_error_penalty,
                                        perp_cmd_threshold=cfg.perp_cmd_threshold)
    cfg.enabled = True
    if agent_cfg.finetune:
        # 与四组微调 A–E 相同的优化设置，便于同起点、同预算对比
        agent_cfg.learning_rate = 3e-5
        agent_cfg.anneal_lr = False

    per_distance = cfg.cost_metric == "per_distance"
    metric = ("q = 平滑功率 / (平滑有效速度 + v_0)，v_0={:.2f} 米/秒".format(cfg.speed_offset)
              if per_distance else "平滑机械功率")
    track = (f"每步全环境水平速度误差均值，<= {cfg.err_tighten_ratio} 倍热身参考值才收紧，> {cfg.err_backoff_ratio} 倍回退"
             if cfg.track_mode == "error" else
             "跟踪奖励 >= perf_ratio*track_ref 才收紧，< (perf_ratio-relax_margin)*track_ref 回退")
    if cfg.drift_tighten_ratio is not None:
        track += (f"；侧滑速度与偏航误差相对热身参考值倍数的较大者 <= {cfg.drift_tighten_ratio} 才收紧，"
                  f"> {cfg.drift_backoff_ratio} 回退")
    v4 = cfg.lateral_penalty > 0 or cfg.yaw_error_penalty > 0
    floor = cfg.lambda_init if cfg.lambda_integral_floor is None else cfg.lambda_integral_floor
    manifest = {
        "约束形式": f"PPO-Lagrangian（拉格朗日对偶 PPO）：E[{metric}] <= 上限（{unit}）",
        "版本": ("第四版（2026-11-04）：有效速度扣除侧滑与偏航误差；窄防抖带；回退不抬过实测值；λ 保底；漂移监控"
                 if per_distance and v4 else
                 "第三版（2026-11-03）：约束每米能耗；有效速度为指令方向投影并截断；λ 初值对齐原奖励权重并预热；速度误差为收紧判据"
                 if per_distance else "第二版（2026-11-01）：约束平滑功率"),
        "有效速度": (f"max(0, v_par + {cfg.yaw_rate_weight}*clip(w_z*sign(w_cmd), 0, |w_cmd|) - {cfg.lateral_penalty}*v_perp "
                     f"- {cfg.yaw_error_penalty}*|w_z - w_cmd|)；v_par 为机体系水平速度在指令方向的投影并截断到 [0, |v_cmd_xy|]，"
                     f"v_perp 为垂直于指令方向的水平速度（|v_cmd_xy| < {cfg.perp_cmd_threshold} 时取整个水平速度）；"
                     "与功率同为 0.5 秒指数滑动平均" if per_distance else "不使用"),
        "回退": ("恢复到上一次收紧前的值" if cfg.backoff_headroom is None else
                 f"恢复到上一次收紧前的值，但不超过 约束量滑动平均*(1+{cfg.backoff_headroom})，也不低于当前上限"),
        "lambda保底": f"热身结束后 λ 及其积分项不低于 {floor}",
        "每步代价": ("(约束量 - 上限) / metric_ref * step_dt，可正可负；热身期以约束量滑动平均为零点"
                     if cfg.cost_centered else "约束量 / metric_ref * step_dt"),
        "metric_ref": (cfg.metric_ref if cfg.metric_ref is not None else
                       ("在 λ 置零预热期末标定为 power_ref / 平均分母" if per_distance else "取 power_ref")),
        "平滑方式": f"指数滑动平均，时间常数 {env_cfg.energy_cost.tau_s} 秒；新回合首步以瞬时值初始化",
        "优势合成": "(A_r - lambda*A_c) / (1 + lambda)",
        "lambda预热": (f"前 {cfg.lambda_warmup_zero_iters} 次为 0（只训练代价网络），之后 {cfg.lambda_warmup_ramp_iters} 次线性升到 "
                       f"{cfg.lambda_init}，保持到第 {cfg.warmup_iters} 次热身结束"),
        "lambda更新": ("积分项 I <- clip(I + lr*v, 0, lambda_max)，v=(约束量均值 - 上限)/metric_ref；"
                       "lambda = clip(I + kp*EMA(v), 0, lambda_max)"),
        "上限调整": (f"{track}；地形 < terrain_floor*terrain_ref 也回退；回退到上一次收紧前的值（每次收紧最多回退一次），"
                     "并暂停收紧 backoff_cooldown 个判断周期；上限不会超过已到达过的最高值"),
        "已移除奖励项": "minimize_power",
        "原能耗奖励终值权重": removed_weight,
        "初始上限说明": "热身结束时用实测约束量滑动平均标定" if cfg.limit_init is None else "命令行指定",
        "学习率": agent_cfg.learning_rate, "学习率退火": agent_cfg.anneal_lr,
        "参数": cfg.to_dict(),
    }
    if reference_info:
        manifest.update(reference_info)
    return manifest
