# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration terms for different managers."""

from __future__ import annotations

import torch
from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.managers.manager_term_cfg import ManagerTermBaseCfg


##
# Constraint manager.
##


@configclass
class ConstraintTermCfg(ManagerTermBaseCfg):
    func: Callable[..., torch.Tensor] = MISSING

    max_p: float = MISSING


@configclass
class EnergyCostCfg:
    """拉格朗日能耗约束的环境侧代价：平滑机械功率 sum|tau*dq| 的指数滑动平均。"""

    tau_s: float = 0.5
    """指数滑动平均的时间常数（秒）。"""

    power_ref: float = MISSING
    """功率归一化尺度（瓦），只用于约束量为功率（第二版）时的代价归一化。"""

    speed_offset: float = 0.28
    """每米能耗分母的速度偏移 v_0（米/秒）：q = 平滑功率 / (平滑有效速度 + v_0)。
    取 A 组平地 0.4/0.8/1.0 米/秒评估数据 P(v)=P_0+k*v 的拟合比值 P_0/k，用于抵消站立与姿态维持的基础功率。"""

    yaw_rate_weight: float = 0.5
    """有效速度中转向项的权重（米/弧度）：v_eff = 指令方向投影速度 + w * 指令方向偏航角速度。"""

    lateral_penalty: float = 0.5
    """有效速度中扣除的侧滑速度权重：v_eff -= 0.5 * 垂直于指令方向的水平速度。0 时与第三版相同。"""

    yaw_error_penalty: float = 0.3
    """有效速度中扣除的偏航角速度误差权重（米/弧度）：v_eff -= 0.3 * |w_z - w_cmd|。0 时与第三版相同。"""

    perp_cmd_threshold: float = 0.1
    """水平指令小于该值（米/秒）时方向不可靠，整个水平速度都计为侧滑。"""
