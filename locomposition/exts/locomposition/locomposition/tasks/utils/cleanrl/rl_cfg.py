# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass


@configclass
class EnergyConstraintCfg:
    """PPO-Lagrangian（拉格朗日对偶 PPO）能耗约束与动态上限。

    默认值为第四版（2026-11-04）：约束每米能耗 q = 平滑功率/(平滑有效速度+v_0)，有效速度扣除侧滑与偏航误差；
    速度误差与漂移为收紧判据；回退不抬过实测值；λ 积分项保底。
    复现第三版需覆盖：lateral_penalty=0 yaw_error_penalty=0 err_tighten_ratio=1.04 err_backoff_ratio=1.09
    drift_tighten_ratio=null drift_backoff_ratio=null backoff_headroom=null lambda_integral_floor=0。
    复现第二版需覆盖：cost_metric=power track_mode=reward lambda_init=0 lambda_warmup_zero_iters=0
    lambda_warmup_ramp_iters=0 limit_min=10 limit_max=80。
    """

    enabled: bool = False
    power_ref: float = 40.05852980620652  # 瓦；gpt-固定参考尺度.json 中基线 model_30000 的 P_ref
    # 约束量："per_distance" 为每米能耗 q（焦耳/米，第三版）；"power" 为平滑机械功率（瓦，第一、二版）
    cost_metric: Literal["per_distance", "power"] = "per_distance"
    speed_offset: float = 0.28  # 米/秒；q 分母的偏移 v_0，见 EnergyCostCfg.speed_offset
    yaw_rate_weight: float = 0.5  # 米/弧度；有效速度中的转向项权重
    # 约束量的归一化尺度（约束量单位）。每步代价 = (约束量 - 上限)/metric_ref*dt，
    # 对固定分母 D，等效的每瓦能耗压力为 λ/(metric_ref*D)；取 metric_ref = power_ref/D 时
    # λ = 0.008*power_ref 与原奖励权重 0.008/瓦 对齐。
    # None：power 模式取 power_ref；per_distance 模式在 λ 为 0 的预热期末（第 lambda_warmup_zero_iters 次）
    # 用实测平均分母 D 自动标定为 power_ref/D。按训练指令分布离线估计 D≈0.90–1.06，metric_ref≈38–44
    metric_ref: float | None = None
    limit_init: float | None = None  # 约束量单位；None 表示热身结束时用实测滑动平均标定
    limit_min: float = 5.0
    limit_max: float = 200.0
    # 表现参考值取自微调 C 组平台期（1000–3000 次迭代）
    terrain_ref: float = 3.2
    track_ref: float = 0.80
    perf_ratio: float = 0.95  # track_mode=reward：跟踪奖励 >= 0.95*track_ref 才允许收紧
    relax_margin: float = 0.05  # track_mode=reward：跟踪奖励 < 0.90*track_ref 时回退一步
    # track_mode=error：每步全环境水平速度误差均值，相对热身末参考值的倍数
    track_mode: Literal["error", "reward"] = "error"
    # 第四版：两门槛之间只留窄防抖带，避免第三版那样卡在 1.04–1.09 的死区里一直不收紧
    err_tighten_ratio: float = 1.09  # 速度误差 <= 1.09 倍参考值才允许收紧
    err_backoff_ratio: float = 1.12  # 速度误差 > 1.12 倍参考值时回退一步
    # 侧滑速度与偏航角速度误差监控：取两者相对热身参考值倍数的较大者；None 关闭
    drift_tighten_ratio: float | None = 1.25
    drift_backoff_ratio: float | None = 1.40
    # 回退后的上限不超过 约束量滑动平均*(1+headroom)，防止上限明显高于实测值、λ 泄掉；None 时按第三版恢复
    backoff_headroom: float | None = 0.02
    # 热身结束后 λ 及其积分项都不低于该值；None 时取 lambda_init（与原奖励权重 0.008/瓦 对齐），0 时与第三版相同
    lambda_integral_floor: float | None = None
    # 有效速度中扣除侧滑与偏航误差（第四版）；两者为 0 时与第三版相同
    lateral_penalty: float = 0.5
    yaw_error_penalty: float = 0.3
    perp_cmd_threshold: float = 0.1
    terrain_floor: float = 0.85  # 地形等级只作下限：< 0.85*terrain_ref 时回退一步，且不收紧
    satisfy_tol: float = 0.05
    shrink: float = 0.03
    adapt_interval: int = 50
    warmup_iters: int = 500
    backoff_cooldown: int = 4  # 回退后暂停收紧的判断周期数（每周期 adapt_interval 次迭代）
    perf_beta: float = 0.9
    # 起点策略训练时的能耗奖励权重为 0.008/瓦；λ 等效的每瓦压力为 λ/(metric_ref*D)，D≈power_ref/metric_ref 时
    # λ = 0.008*power_ref ≈ 0.32 与之对齐，消除奖励切换为约束时的空档
    lambda_init: float = 0.32
    lambda_warmup_zero_iters: int = 100  # 前 100 次 λ=0，只训练代价网络
    lambda_warmup_ramp_iters: int = 100  # 之后 100 次 λ 线性升到 lambda_init，保持到热身结束
    lambda_lr: float = 0.05
    lambda_kp: float = 10.0  # 比例项增益，作用于违反量的指数滑动平均
    lambda_kp_beta: float = 0.9
    lambda_max: float = 10.0
    # True：每步代价为 (约束量 - 上限)/metric_ref*dt，可正可负，消除靠提前终止逃避代价的动机；
    # False：第一版的恒正代价 约束量/metric_ref*dt
    cost_centered: bool = True
    cost_critic_lr: float = 3.0e-4


@configclass
class CleanRlPpoActorCriticCfg:
    seed: int = MISSING # WILL BE OVERWRITTEN IN train.py

    save_interval: int = MISSING

    learning_rate: float = MISSING
    num_steps: int = MISSING
    num_iterations: int = MISSING
    gamma: float = MISSING
    gae_lambda: float = MISSING
    updates_epochs: int = MISSING
    minibatch_size: int = MISSING
    clip_coef: float = MISSING
    ent_coef: float = MISSING
    vf_coef: float = MISSING
    max_grad_norm: float = MISSING
    norm_adv: bool = MISSING
    clip_vloss: bool = MISSING
    anneal_lr: bool = MISSING

    experiment_name: str = MISSING
    logger: Literal["tensorboard", "wandb"] = "tensorboard"
    wandb_project: str = MISSING

    load_run: str = MISSING
    load_checkpoint: str = MISSING
    resume: bool = False
    finetune: bool = False
    allow_partial_resume: bool = False
    resume_constraint_progress: float = 1.0
    stop_after_iterations: int = 0
    energy_constraint: EnergyConstraintCfg = EnergyConstraintCfg()
