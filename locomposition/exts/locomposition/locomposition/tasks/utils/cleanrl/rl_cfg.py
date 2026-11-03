# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass


@configclass
class EnergyConstraintCfg:
    """PPO-Lagrangian（拉格朗日对偶 PPO）能耗约束与动态功率上限。"""

    enabled: bool = False
    power_ref: float = 40.05852980620652  # 瓦；gpt-固定参考尺度.json 中基线 model_30000 的 P_ref
    limit_init: float | None = None  # 瓦；None 表示热身结束时用实测平滑功率标定
    limit_min: float = 10.0
    limit_max: float = 80.0
    # 表现参考值取自微调 C 组平台期（1000–3000 次迭代）
    terrain_ref: float = 3.2
    track_ref: float = 0.80
    perf_ratio: float = 0.95  # 跟踪奖励 >= 0.95*track_ref 才允许收紧
    relax_margin: float = 0.05  # 跟踪奖励 < 0.90*track_ref 时回退一步
    terrain_floor: float = 0.85  # 地形等级只作下限：< 0.85*terrain_ref 时回退一步，且不收紧
    satisfy_tol: float = 0.05
    shrink: float = 0.03
    adapt_interval: int = 50
    warmup_iters: int = 500
    backoff_cooldown: int = 4  # 回退后暂停收紧的判断周期数（每周期 adapt_interval 次迭代）
    perf_beta: float = 0.9
    lambda_init: float = 0.0
    lambda_lr: float = 0.05
    lambda_kp: float = 10.0  # 比例项增益，作用于违反量的指数滑动平均
    lambda_kp_beta: float = 0.9
    lambda_max: float = 10.0
    # True：每步代价为 (平滑功率 - P_lim)/power_ref*dt，可正可负，消除靠提前终止逃避代价的动机；
    # False：第一版的恒正代价 平滑功率/power_ref*dt
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
