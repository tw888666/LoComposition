"""能耗拉格朗日约束：乘子更新与按表现自适应收紧功率上限。

纯 Python 实现，不依赖 Isaac Sim（仿真器），便于离线单测。
约束形式：E[平滑机械功率] <= P_lim，违反量按 P_ref 归一化。

第二版（2026-11-01）针对第一版正式训练中上限与 λ 两轮振荡、性能崩溃的问题：
上限只收紧或保持且最多回退一步；λ 加比例项；跟踪为主判据、地形只作下限。
第四版（2026-11-04）针对第三版调度卡死在死区、回退后 λ 泄掉、检查点 1049 横向漂移：
收紧门槛与回退门槛之间只留窄防抖带；回退后上限不明显高于实测约束量；λ 积分项保底；
另加侧滑与偏航误差监控。v_eff 中扣除侧滑与偏航误差在 cat_env.py 中实现。
"""
import math


def _clamp(value, low, high):
    return min(max(value, low), high)


class LagrangeMultiplier:
    """PI 形式的拉格朗日乘子（Stooke 等 2020 的 PID 拉格朗日去掉微分项）。

    积分项为标准对偶梯度上升；比例项作用于违反量的指数滑动平均，使 λ 在超限与恢复两个方向都能快速响应，
    同时不被单次迭代的功率尖峰带偏。kp=0 时退化为标准形式。
    """

    def __init__(self, *, init, lr, kp=0.0, kp_beta=0.9, max_value=10.0, integral_floor=0.0):
        if not (init >= 0 and lr > 0 and kp >= 0 and 0 <= kp_beta < 1 and max_value > 0
                and 0 <= integral_floor <= max_value):
            raise ValueError("拉格朗日乘子参数须满足 init>=0, lr>0, kp>=0, 0<=kp_beta<1, max_value>0, 0<=integral_floor<=max_value。")
        self.lr = float(lr)
        self.kp = float(kp)
        self.kp_beta = float(kp_beta)
        self.max_value = float(max_value)
        # 保底：热身结束后积分项与 λ 本身都不低于它，防止约束暂时满足时 λ 一路泄掉、能耗反弹。
        # 只保底积分项不够：kp=10 时违反量滑动平均为 -0.05 就能让比例项把 λ 压到 0。
        # 比例项只能把 λ 往上推高于保底，不能压到保底以下
        self.integral_floor = float(integral_floor)
        self.init = _clamp(float(init), 0.0, self.max_value)
        self.integral = self.init
        self.violation_ema = 0.0
        self.value = self.integral

    def warmup_value(self, iteration, *, zero_iters, ramp_iters):
        """热身期 λ：前 zero_iters 次为 0（代价网络预热），随后 ramp_iters 次线性升到 init；不改变内部状态。"""
        if iteration <= zero_iters:
            return 0.0
        if ramp_iters <= 0:
            return self.init
        return self.init * min((iteration - zero_iters) / ramp_iters, 1.0)

    def set_warmup_value(self, value):
        """热身期直接设定 λ，积分项同步，热身结束后从这里继续对偶更新。"""
        self.integral = _clamp(float(value), 0.0, self.max_value)
        self.value = self.integral

    def update(self, violation):
        """violation=(平均功率-上限)/P_ref；正数表示超限，λ 上升。"""
        if not math.isfinite(violation):
            raise ValueError(f"约束违反量非有限数值：{violation}")
        self.integral = _clamp(self.integral + self.lr * violation, self.integral_floor, self.max_value)
        self.violation_ema = self.kp_beta * self.violation_ema + (1.0 - self.kp_beta) * violation
        self.value = _clamp(self.integral + self.kp * self.violation_ema, self.integral_floor, self.max_value)
        return self.value

    def state_dict(self):
        return {"integral": self.integral, "value": self.value, "violation_ema": self.violation_ema}

    def load_state_dict(self, state):
        self.integral = float(state["integral"])
        self.value = float(state["value"])
        # 第一版检查点没有比例项状态
        self.violation_ema = float(state.get("violation_ema", 0.0))


class AdaptiveLimitScheduler:
    """按表现调整约束上限，只收紧或保持；表现下降时最多回退一步并暂停收紧。

    主判据为速度跟踪（反应快），地形等级只作下限（反应慢，只用于拦截明显退化）：
    - 收紧：跟踪达标、地形 >= terrain_floor*terrain_ref、且约束满足；
    - 回退：跟踪明显变差或地形低于下限，上限恢复到最近一次收紧前的值，
      每次收紧只能被回退一次，之后暂停收紧 cooldown 个判断周期；
    - 其余情况保持不变。上限永远不会超过已到达过的最高值，因此不会连续放松。
    跟踪判据有两种：track_mode="reward" 用跟踪奖励（第二版），"error" 用水平速度误差相对热身参考值的倍数（第三版）。
    参数名中的 power 沿用第二版，第三、四版中它是每米能耗约束量 q（焦耳/米）。
    第四版可选开启漂移监控：侧滑速度或偏航误差超过参考值倍数时同样回退、不收紧。
    """

    def __init__(self, *, limit_init, limit_min, limit_max, terrain_ref, track_ref,
                 perf_ratio=0.95, relax_margin=0.05, terrain_floor=0.85, satisfy_tol=0.05,
                 shrink=0.03, interval=50, warmup=500, cooldown=4, beta=0.9,
                 track_mode="reward", err_tighten_ratio=1.04, err_backoff_ratio=1.09,
                 backoff_headroom=None, drift_tighten_ratio=None, drift_backoff_ratio=None):
        # limit_init=None：热身结束时以实测约束量的滑动平均作为初始上限。
        # track_mode="reward"：track 为跟踪奖励（越大越好），参考值 track_ref 固定；
        # track_mode="error"：track 为水平速度误差（越小越好），参考值在热身末自动标定为其滑动平均。
        # backoff_headroom：回退后的上限不超过 约束量滑动平均*(1+headroom)，避免上限明显高于实测值而让 λ 泄掉；
        #   None 时直接恢复到上一次收紧前的值（第二、三版）。
        # drift_tighten_ratio / drift_backoff_ratio：侧滑速度与偏航角速度误差相对热身参考值的倍数门槛
        #   （取两者中较大的倍数），None 时不监控（第二、三版）。
        if not 0 < limit_min <= limit_max or (limit_init is not None and not limit_min <= limit_init <= limit_max):
            raise ValueError("功率上限须满足 0 < limit_min <= limit_init <= limit_max。")
        self.drift_enabled = drift_tighten_ratio is not None or drift_backoff_ratio is not None
        if (limit_init is None or track_mode == "error" or self.drift_enabled) and warmup < 1:
            raise ValueError("自动标定初始上限、速度误差或漂移参考值需要至少一次热身迭代。")
        if track_mode not in ("reward", "error"):
            raise ValueError(f"track_mode 只能是 reward 或 error，收到 {track_mode!r}。")
        if not 1 <= err_tighten_ratio < err_backoff_ratio:
            raise ValueError("速度误差门槛须满足 1 <= err_tighten_ratio < err_backoff_ratio。")
        if self.drift_enabled and not (drift_tighten_ratio is not None and drift_backoff_ratio is not None
                                       and 1 <= drift_tighten_ratio < drift_backoff_ratio):
            raise ValueError("漂移门槛须同时给出，且满足 1 <= drift_tighten_ratio < drift_backoff_ratio。")
        if backoff_headroom is not None and not backoff_headroom >= 0:
            raise ValueError("backoff_headroom 不得为负。")
        if not (terrain_ref > 0 and track_ref > 0 and 0 < perf_ratio <= 1 and 0 <= relax_margin < perf_ratio
                and 0 < terrain_floor <= 1):
            raise ValueError("表现参考值须为正，perf_ratio 位于 (0,1]，relax_margin 位于 [0, perf_ratio)，terrain_floor 位于 (0,1]。")
        if not (0 < shrink < 1 and satisfy_tol >= 0 and interval >= 1 and warmup >= 0 and cooldown >= 0 and 0 <= beta < 1):
            raise ValueError("收紧比例、容差、间隔、热身、暂停周期或平滑系数不合法。")
        self.limit = None if limit_init is None else float(limit_init)
        self.limit_min = float(limit_min)
        self.limit_max = float(limit_max)
        self.terrain_ref = float(terrain_ref)
        self.track_ref = float(track_ref)
        self.perf_ratio = float(perf_ratio)
        self.relax_margin = float(relax_margin)
        self.terrain_floor = float(terrain_floor)
        self.satisfy_tol = float(satisfy_tol)
        self.shrink = float(shrink)
        self.interval = int(interval)
        self.warmup = int(warmup)
        self.cooldown = int(cooldown)
        self.beta = float(beta)
        self.track_mode = track_mode
        self.err_tighten_ratio = float(err_tighten_ratio)
        self.err_backoff_ratio = float(err_backoff_ratio)
        self.error_ref = None  # track_mode="error" 时在热身末标定
        self.backoff_headroom = None if backoff_headroom is None else float(backoff_headroom)
        self.drift_tighten_ratio = None if drift_tighten_ratio is None else float(drift_tighten_ratio)
        self.drift_backoff_ratio = None if drift_backoff_ratio is None else float(drift_backoff_ratio)
        self.lateral_ref = None  # 侧滑速度参考值，漂移监控开启时在热身末标定
        self.yaw_error_ref = None  # 偏航角速度误差参考值
        # 热身后半段 约束量、跟踪量、侧滑、偏航误差 之和；只在热身期使用，不进检查点
        self._calib_sum = [0.0, 0.0, 0.0, 0.0]
        self._calib_count = 0
        self.iterations = 0
        self.terrain_polyak = None
        self.track_polyak = None
        self.power_polyak = None
        self.lateral_polyak = None
        self.yaw_error_polyak = None
        self.previous_limit = None  # 最近一次收紧前的上限；回退后置空，保证同一步只回退一次
        self.cooldown_left = 0
        # 最近一次判断时的倍数，只用于日志
        self.last_error_ratio = float("nan")
        self.last_drift_ratio = float("nan")

    @property
    def in_warmup(self):
        return self.iterations <= self.warmup

    def _polyak(self, old, new):
        return new if old is None else self.beta * old + (1.0 - self.beta) * new

    def observe(self, *, terrain, track, power, lateral=None, yaw_error=None):
        """每次迭代调用一次；返回本次决策："warmup"、"hold"、"tighten" 或 "backoff"。

        lateral、yaw_error 为侧滑速度与偏航角速度误差的全环境均值，只在开启漂移监控时需要。
        """
        inputs = [("terrain", terrain), ("track", track), ("power", power)]
        if self.drift_enabled:
            if lateral is None or yaw_error is None:
                raise ValueError("开启漂移监控时必须提供 lateral 与 yaw_error。")
            inputs += [("lateral", lateral), ("yaw_error", yaw_error)]
        for name, value in inputs:
            if not math.isfinite(value):
                raise ValueError(f"自适应上限输入 {name} 非有限数值：{value}")
        self.iterations += 1
        self.terrain_polyak = self._polyak(self.terrain_polyak, terrain)
        self.track_polyak = self._polyak(self.track_polyak, track)
        self.power_polyak = self._polyak(self.power_polyak, power)
        if self.drift_enabled:
            self.lateral_polyak = self._polyak(self.lateral_polyak, lateral)
            self.yaw_error_polyak = self._polyak(self.yaw_error_polyak, yaw_error)
        if self.in_warmup:
            # 标定只用热身后半段的算术平均：开头所有环境同时从静止起步，约束量与速度误差偏大
            if self.iterations > self.warmup // 2:
                self._calib_sum[0] += power
                self._calib_sum[1] += track
                if self.drift_enabled:
                    self._calib_sum[2] += lateral
                    self._calib_sum[3] += yaw_error
                self._calib_count += 1
            if self.iterations == self.warmup:
                mean_power, mean_track, mean_lateral, mean_yaw = (s / self._calib_count for s in self._calib_sum)
                if self.limit is None:
                    self.limit = _clamp(mean_power, self.limit_min, self.limit_max)
                if self.track_mode == "error" and self.error_ref is None:
                    if not mean_track > 0:
                        raise ValueError(f"热身后半段速度误差均值必须为正，收到 {mean_track}。")
                    self.error_ref = mean_track
                if self.drift_enabled and self.lateral_ref is None:
                    if not (mean_lateral > 0 and mean_yaw > 0):
                        raise ValueError(f"热身后半段侧滑与偏航误差均值必须为正，收到 {mean_lateral}、{mean_yaw}。")
                    self.lateral_ref, self.yaw_error_ref = mean_lateral, mean_yaw
                    self.lateral_polyak, self.yaw_error_polyak = mean_lateral, mean_yaw
                # 判断用的滑动平均从标定值重新起步，避免继承开头的瞬态
                self.power_polyak, self.track_polyak = mean_power, mean_track
            return "warmup"
        if (self.iterations - self.warmup) % self.interval != 0:
            return "hold"
        terrain_ratio = self.terrain_polyak / self.terrain_ref
        terrain_ok = terrain_ratio >= self.terrain_floor
        if self.track_mode == "error":
            error_ratio = self.track_polyak / self.error_ref
            track_good = error_ratio <= self.err_tighten_ratio
            track_bad = error_ratio > self.err_backoff_ratio
            self.last_error_ratio = error_ratio
        else:
            track_ratio = self.track_polyak / self.track_ref
            track_good = track_ratio >= self.perf_ratio
            track_bad = track_ratio < self.perf_ratio - self.relax_margin
        drift_good, drift_bad = True, False
        if self.drift_enabled:
            drift_ratio = max(self.lateral_polyak / self.lateral_ref, self.yaw_error_polyak / self.yaw_error_ref)
            drift_good = drift_ratio <= self.drift_tighten_ratio
            drift_bad = drift_ratio > self.drift_backoff_ratio
            self.last_drift_ratio = drift_ratio
        performance_bad = track_bad or not terrain_ok or drift_bad
        if performance_bad:
            # 回退到最近一次收紧前的值，并暂停收紧；已回退过则只保持
            self.cooldown_left = self.cooldown
            if self.previous_limit is not None:
                target = self.previous_limit
                if self.backoff_headroom is not None:
                    # 不抬到明显高于实测约束量，但也不低于当前上限（回退不能变成收紧）
                    target = min(target, max(self.limit, self.power_polyak * (1.0 + self.backoff_headroom)))
                self.limit = target
                self.previous_limit = None
                return "backoff"
            return "hold"
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
            return "hold"
        satisfied = self.power_polyak <= self.limit * (1.0 + self.satisfy_tol)
        if track_good and drift_good and terrain_ok and satisfied and self.limit > self.limit_min:
            self.previous_limit = self.limit
            self.limit = max(self.limit_min, self.limit * (1.0 - self.shrink))
            return "tighten"
        return "hold"

    def state_dict(self):
        return {name: getattr(self, name) for name in
                ("limit", "iterations", "terrain_polyak", "track_polyak", "power_polyak",
                 "previous_limit", "cooldown_left", "error_ref",
                 "lateral_ref", "yaw_error_ref", "lateral_polyak", "yaw_error_polyak")}

    def load_state_dict(self, state):
        self.limit = None if state["limit"] is None else float(state["limit"])
        self.iterations = int(state["iterations"])
        for name in ("terrain_polyak", "track_polyak", "power_polyak"):
            value = state[name]
            setattr(self, name, None if value is None else float(value))
        # 第一版检查点没有回退与暂停状态
        previous = state.get("previous_limit")
        self.previous_limit = None if previous is None else float(previous)
        self.cooldown_left = int(state.get("cooldown_left", 0))
        # 第二版及以前的检查点没有速度误差参考值；第三版及以前没有漂移监控状态
        for name in ("error_ref", "lateral_ref", "yaw_error_ref", "lateral_polyak", "yaw_error_polyak"):
            value = state.get(name)
            setattr(self, name, None if value is None else float(value))
