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
    """功率归一化尺度（瓦）；每步代价为 平滑功率 / power_ref * step_dt。"""
