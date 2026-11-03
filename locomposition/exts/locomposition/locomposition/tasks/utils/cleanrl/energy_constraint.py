"""能耗拉格朗日约束：乘子更新与按表现自适应收紧功率上限。

纯 Python 实现，不依赖 Isaac Sim（仿真器），便于离线单测。
约束形式：E[平滑机械功率] <= P_lim，违反量按 P_ref 归一化。

第二版（2026-11-01）针对第一版正式训练中上限与 λ 两轮振荡、性能崩溃的问题：
上限只收紧或保持且最多回退一步；λ 加比例项；跟踪为主判据、地形只作下限。
"""
import math


def _clamp(value, low, high):
    return min(max(value, low), high)


class LagrangeMultiplier:
    """PI 形式的拉格朗日乘子（Stooke 等 2020 的 PID 拉格朗日去掉微分项）。

    积分项为标准对偶梯度上升；比例项作用于违反量的指数滑动平均，使 λ 在超限与恢复两个方向都能快速响应，
    同时不被单次迭代的功率尖峰带偏。kp=0 时退化为标准形式。
    """

    def __init__(self, *, init, lr, kp=0.0, kp_beta=0.9, max_value=10.0):
        if not (init >= 0 and lr > 0 and kp >= 0 and 0 <= kp_beta < 1 and max_value > 0):
            raise ValueError("拉格朗日乘子参数须满足 init>=0, lr>0, kp>=0, 0<=kp_beta<1, max_value>0。")
        self.lr = float(lr)
        self.kp = float(kp)
        self.kp_beta = float(kp_beta)
        self.max_value = float(max_value)
        self.integral = _clamp(float(init), 0.0, self.max_value)
        self.violation_ema = 0.0
        self.value = self.integral

    def update(self, violation):
        """violation=(平均功率-上限)/P_ref；正数表示超限，λ 上升。"""
        if not math.isfinite(violation):
            raise ValueError(f"约束违反量非有限数值：{violation}")
        self.integral = _clamp(self.integral + self.lr * violation, 0.0, self.max_value)
        self.violation_ema = self.kp_beta * self.violation_ema + (1.0 - self.kp_beta) * violation
        self.value = _clamp(self.integral + self.kp * self.violation_ema, 0.0, self.max_value)
        return self.value

    def state_dict(self):
        return {"integral": self.integral, "value": self.value, "violation_ema": self.violation_ema}

    def load_state_dict(self, state):
        self.integral = float(state["integral"])
        self.value = float(state["value"])
        # 第一版检查点没有比例项状态
        self.violation_ema = float(state.get("violation_ema", 0.0))


class AdaptiveLimitScheduler:
    """按表现调整功率上限，只收紧或保持；表现下降时最多回退一步并暂停收紧。

    主判据为速度跟踪（反应快），地形等级只作下限（反应慢，只用于拦截明显退化）：
    - 收紧：跟踪 >= perf_ratio*track_ref、地形 >= terrain_floor*terrain_ref、且约束满足；
    - 回退：跟踪 < (perf_ratio-relax_margin)*track_ref 或地形低于下限，上限恢复到最近一次收紧前的值，
      每次收紧只能被回退一次，之后暂停收紧 cooldown 个判断周期；
    - 其余情况保持不变。上限永远不会超过已到达过的最高值，因此不会连续放松。
    """

    def __init__(self, *, limit_init, limit_min, limit_max, terrain_ref, track_ref,
                 perf_ratio=0.95, relax_margin=0.05, terrain_floor=0.85, satisfy_tol=0.05,
                 shrink=0.03, interval=50, warmup=500, cooldown=4, beta=0.9):
        # limit_init=None：热身结束时以实测平滑功率的滑动平均作为初始上限。
        if not 0 < limit_min <= limit_max or (limit_init is not None and not limit_min <= limit_init <= limit_max):
            raise ValueError("功率上限须满足 0 < limit_min <= limit_init <= limit_max。")
        if limit_init is None and warmup < 1:
            raise ValueError("自动标定初始上限需要至少一次热身迭代。")
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
        self.iterations = 0
        self.terrain_polyak = None
        self.track_polyak = None
        self.power_polyak = None
        self.previous_limit = None  # 最近一次收紧前的上限；回退后置空，保证同一步只回退一次
        self.cooldown_left = 0

    @property
    def in_warmup(self):
        return self.iterations <= self.warmup

    def _polyak(self, old, new):
        return new if old is None else self.beta * old + (1.0 - self.beta) * new

    def observe(self, *, terrain, track, power):
        """每次迭代调用一次；返回本次决策："warmup"、"hold"、"tighten" 或 "relax"。"""
        for name, value in (("terrain", terrain), ("track", track), ("power", power)):
            if not math.isfinite(value):
                raise ValueError(f"自适应上限输入 {name} 非有限数值：{value}")
        self.iterations += 1
        self.terrain_polyak = self._polyak(self.terrain_polyak, terrain)
        self.track_polyak = self._polyak(self.track_polyak, track)
        self.power_polyak = self._polyak(self.power_polyak, power)
        if self.in_warmup:
            if self.iterations == self.warmup and self.limit is None:
                self.limit = _clamp(self.power_polyak, self.limit_min, self.limit_max)
            return "warmup"
        if (self.iterations - self.warmup) % self.interval != 0:
            return "hold"
        terrain_ratio = self.terrain_polyak / self.terrain_ref
        track_ratio = self.track_polyak / self.track_ref
        terrain_ok = terrain_ratio >= self.terrain_floor
        performance_bad = track_ratio < self.perf_ratio - self.relax_margin or not terrain_ok
        if performance_bad:
            # 回退到最近一次收紧前的值，并暂停收紧；已回退过则只保持
            self.cooldown_left = self.cooldown
            if self.previous_limit is not None:
                self.limit = self.previous_limit
                self.previous_limit = None
                return "backoff"
            return "hold"
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
            return "hold"
        satisfied = self.power_polyak <= self.limit * (1.0 + self.satisfy_tol)
        if track_ratio >= self.perf_ratio and terrain_ok and satisfied and self.limit > self.limit_min:
            self.previous_limit = self.limit
            self.limit = max(self.limit_min, self.limit * (1.0 - self.shrink))
            return "tighten"
        return "hold"

    def state_dict(self):
        return {name: getattr(self, name) for name in
                ("limit", "iterations", "terrain_polyak", "track_polyak", "power_polyak",
                 "previous_limit", "cooldown_left")}

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
