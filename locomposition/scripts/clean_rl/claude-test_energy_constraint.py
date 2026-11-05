"""能耗拉格朗日约束的离线单测（不启动 Isaac Sim）。运行：python -I scripts/clean_rl/claude-test_energy_constraint.py"""
import importlib.util
import math
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[2] / "exts/locomposition/locomposition/tasks/utils/cleanrl/energy_constraint.py"
spec = importlib.util.spec_from_file_location("energy_constraint", MODULE_PATH)
ec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ec)


def test_lagrange_rises_and_falls():
    lam = ec.LagrangeMultiplier(init=0.0, lr=0.5, max_value=2.0)
    for _ in range(3):
        lam.update(0.2)
    assert math.isclose(lam.value, 0.3), lam.value
    for _ in range(10):
        lam.update(-0.2)
    assert lam.value == 0.0
    for _ in range(100):
        lam.update(1.0)
    assert lam.value == 2.0
    restored = ec.LagrangeMultiplier(init=0.0, lr=0.5)
    restored.load_state_dict(lam.state_dict())
    assert restored.value == lam.value


def test_lagrange_proportional_term():
    lam = ec.LagrangeMultiplier(init=0.0, lr=0.1, kp=1.0, kp_beta=0.0)
    lam.update(0.5)
    assert math.isclose(lam.integral, 0.05) and math.isclose(lam.value, 0.55)
    # 违反量转负时比例项立刻把 λ 拉低，积分项尚未归零
    lam.update(-0.3)
    assert math.isclose(lam.integral, 0.02) and lam.value == 0.0


def test_lagrange_proportional_ema_and_legacy_state():
    lam = ec.LagrangeMultiplier(init=0.0, lr=0.1, kp=2.0, kp_beta=0.5)
    lam.update(1.0)
    assert math.isclose(lam.violation_ema, 0.5) and math.isclose(lam.value, 0.1 + 2.0 * 0.5)
    restored = ec.LagrangeMultiplier(init=0.0, lr=0.1, kp=2.0, kp_beta=0.5)
    restored.load_state_dict(lam.state_dict())
    assert restored.state_dict() == lam.state_dict()
    legacy = ec.LagrangeMultiplier(init=0.0, lr=0.1)
    legacy.load_state_dict({"integral": 0.3, "value": 0.3})  # 第一版检查点
    assert legacy.violation_ema == 0.0


def make_scheduler(**kw):
    args = dict(limit_init=40.0, limit_min=20.0, limit_max=60.0, terrain_ref=3.2, track_ref=0.8,
                interval=2, warmup=2, beta=0.0, shrink=0.1, cooldown=1)
    args.update(kw)
    return ec.AdaptiveLimitScheduler(**args)


def decide(s, *, terrain=3.2, track=0.8, power=30.0):
    """跳过非决策迭代，返回下一次判断的决策。"""
    assert s.observe(terrain=terrain, track=track, power=power) == "hold"
    return s.observe(terrain=terrain, track=track, power=power)


def test_scheduler_tighten_backoff_cooldown():
    s = make_scheduler()
    assert s.observe(terrain=3.2, track=0.8, power=30.0) == "warmup"
    assert s.observe(terrain=3.2, track=0.8, power=30.0) == "warmup"
    assert decide(s) == "tighten" and math.isclose(s.limit, 36.0)
    assert decide(s, power=50.0) == "hold"  # 约束未满足，不收紧
    assert decide(s) == "tighten" and math.isclose(s.limit, 32.4)
    assert decide(s, track=0.6) == "backoff" and math.isclose(s.limit, 36.0)  # 回到上一次收紧前
    assert decide(s, track=0.6) == "hold" and math.isclose(s.limit, 36.0)  # 同一步不再回退，不连续放松
    assert decide(s) == "hold"  # 暂停收紧 1 个判断周期
    assert decide(s) == "tighten" and math.isclose(s.limit, 32.4)


def test_scheduler_terrain_is_floor_only():
    s = make_scheduler(warmup=0)
    # 地形 0.875 倍：低于第一版 0.95 门槛，但高于下限 0.85，跟踪达标即可收紧
    assert decide(s, terrain=2.8) == "tighten"
    # 跟踪 0.92 倍：介于收紧与回退之间，保持
    assert decide(s, track=0.74) == "hold"
    # 地形低于下限：即使跟踪达标也回退
    assert decide(s, terrain=2.6) == "backoff" and math.isclose(s.limit, 40.0)


def test_scheduler_never_exceeds_reached_maximum():
    """复现第一版失败序列：表现持续变差时，上限不能被连续放松到远高于实际功率。"""
    s = make_scheduler(warmup=0, cooldown=0)
    for _ in range(5):
        decide(s)
    peak = 40.0
    for _ in range(30):
        decide(s, terrain=1.0, track=0.4)
        assert s.limit <= peak
    assert s.limit < peak


def test_scheduler_bounds_and_calibration():
    s = make_scheduler(limit_init=None, shrink=0.5, cooldown=0)
    s.observe(terrain=3.2, track=0.8, power=100.0)
    s.observe(terrain=3.2, track=0.8, power=25.0)
    assert s.limit == 25.0  # 热身末用后半段实测均值标定，前半段的 100 被排除
    for _ in range(10):
        s.observe(terrain=3.2, track=0.8, power=1.0)
    assert s.limit == 20.0  # 不低于下界
    restored = make_scheduler()
    restored.load_state_dict(s.state_dict())
    assert restored.state_dict() == s.state_dict()
    legacy = make_scheduler()
    legacy.load_state_dict({"limit": 30.0, "iterations": 7, "terrain_polyak": 3.0,
                            "track_polyak": 0.8, "power_polyak": 31.0})  # 第一版检查点
    assert legacy.previous_limit is None and legacy.cooldown_left == 0


def test_lagrange_warmup_ramp():
    lam = ec.LagrangeMultiplier(init=0.32, lr=0.05)
    values = [lam.warmup_value(i, zero_iters=100, ramp_iters=100) for i in (1, 100, 101, 150, 200, 300)]
    assert values[:2] == [0.0, 0.0]
    assert math.isclose(values[2], 0.0032) and math.isclose(values[3], 0.16)
    assert math.isclose(values[4], 0.32) and math.isclose(values[5], 0.32)
    lam.set_warmup_value(0.16)
    assert lam.integral == lam.value == 0.16
    lam.update(0.0)  # 热身结束后从预热值继续对偶更新
    assert math.isclose(lam.value, 0.16)
    assert lam.warmup_value(5, zero_iters=0, ramp_iters=0) == 0.32


def test_calibration_ignores_startup_transient():
    """复现冒烟中的起步瞬态：热身前半段约束量与速度误差偏大，标定只取后半段均值。"""
    s = make_scheduler(limit_init=None, track_mode="error", warmup=4, beta=0.9, limit_max=200.0)
    for power, track in ((107.2, 0.358), (58.4, 0.172), (43.3, 0.161), (40.2, 0.154)):
        s.observe(terrain=3.2, track=track, power=power)
    assert math.isclose(s.limit, (43.3 + 40.2) / 2) and math.isclose(s.error_ref, (0.161 + 0.154) / 2)
    assert s.power_polyak == s.limit and s.track_polyak == s.error_ref  # 判断用滑动平均从标定值起步


def test_scheduler_error_mode():
    s = make_scheduler(track_mode="error", err_tighten_ratio=1.04, err_backoff_ratio=1.09)
    s.observe(terrain=3.2, track=0.20, power=30.0)
    assert s.observe(terrain=3.2, track=0.20, power=30.0) == "warmup"
    assert math.isclose(s.error_ref, 0.20)  # 热身末标定速度误差参考值
    assert decide(s, track=0.207) == "tighten"  # 1.035 倍：允许收紧
    assert decide(s, track=0.212) == "hold"  # 1.06 倍：介于两门槛之间
    assert decide(s, track=0.220) == "backoff"  # 1.10 倍：回退
    restored = make_scheduler(track_mode="error")
    restored.load_state_dict(s.state_dict())
    assert restored.error_ref == s.error_ref


def test_per_distance_metric_flattens_a_group():
    """用 A 组平地评估数据核对 q = P/(v+v_0)：三个速度下应拉平到同一量级（Argon 方案给出 26~27 焦耳/米）。"""
    data = {0.4: (16.07, 0.329), 0.8: (26.48, 0.720), 1.0: (32.62, 0.914)}
    q = {cmd: p / (v + 0.28) for cmd, (p, v) in data.items()}
    assert all(26.0 < value < 27.5 for value in q.values()), q
    assert (max(q.values()) - min(q.values())) / min(q.values()) < 0.05
    raw = {cmd: p / v for cmd, (p, v) in data.items()}  # 不加偏移时低速明显偏高
    assert raw[0.4] / raw[1.0] > 1.3


def effective_speed(cmd_xy, vel_xy, cmd_yaw, yaw, yaw_weight=0.5, lateral_penalty=0.0, yaw_error_penalty=0.0,
                    perp_threshold=0.1):
    """与 cat_env.py 中相同的有效速度定义，纯 Python 版本；两个扣除项为 0 时即第三版。"""
    speed = math.hypot(*cmd_xy)
    ux, uy = cmd_xy[0] / (speed + 1e-6), cmd_xy[1] / (speed + 1e-6)
    projected = vel_xy[0] * ux + vel_xy[1] * uy
    linear = min(max(projected, 0.0), speed) if speed > 1e-3 else 0.0
    sign = (cmd_yaw > 0) - (cmd_yaw < 0)
    turning = min(max(yaw * sign, 0.0), abs(cmd_yaw))
    if speed < perp_threshold:
        lateral = math.hypot(*vel_xy)
    else:
        lateral = math.hypot(vel_xy[0] - projected * ux, vel_xy[1] - projected * uy)
    return max(0.0, linear + yaw_weight * turning - lateral_penalty * lateral - yaw_error_penalty * abs(yaw - cmd_yaw))


def test_effective_speed_clipping():
    assert math.isclose(effective_speed((1.0, 0.0), (1.5, 0.0), 0.0, 0.0), 1.0, abs_tol=1e-5)  # 超速截断
    assert effective_speed((1.0, 0.0), (-0.5, 0.0), 0.0, 0.0) == 0.0  # 反向不计
    assert math.isclose(effective_speed((1.0, 0.0), (0.6, 0.8), 0.0, 0.0), 0.6, abs_tol=1e-5)  # 侧偏只计投影
    assert math.isclose(effective_speed((0.0, 0.0), (0.3, 0.3), 0.6, 0.4), 0.2)  # 原地转向
    assert math.isclose(effective_speed((0.0, 0.0), (0.0, 0.0), -0.6, 0.4), 0.0)  # 反向转向不计
    assert math.isclose(effective_speed((0.0, 0.0), (0.0, 0.0), 0.6, 1.0), 0.3)  # 转向超速截断


def test_effective_speed_v4_penalties():
    v4 = dict(lateral_penalty=0.5, yaw_error_penalty=0.3)
    # 走直线、偏航对准：与第三版相同
    assert math.isclose(effective_speed((1.0, 0.0), (0.9, 0.0), 0.0, 0.0, **v4), 0.9, abs_tol=1e-5)
    # 侧滑 0.2：扣 0.1
    assert math.isclose(effective_speed((1.0, 0.0), (0.9, 0.2), 0.0, 0.0, **v4), 0.8, abs_tol=1e-5)
    # 偏航误差 0.1 弧度/秒：扣 0.03
    assert math.isclose(effective_speed((1.0, 0.0), (0.9, 0.0), 0.0, 0.1, **v4), 0.87, abs_tol=1e-5)
    # 指令很小：整个水平速度都算侧滑；原地转向时乱走会被扣除
    assert math.isclose(effective_speed((0.05, 0.0), (0.0, 0.2), 0.0, 0.0, **v4), 0.05 * 0.0 + 0.0, abs_tol=1e-5)
    assert math.isclose(effective_speed((0.0, 0.0), (0.1, 0.0), 0.6, 0.6, **v4), 0.3 - 0.05, abs_tol=1e-5)
    # 扣除后不为负
    assert effective_speed((1.0, 0.0), (0.0, 1.0), 0.0, 1.0, **v4) == 0.0


def test_v4_penalties_on_real_trajectories():
    """用评估轨迹核对扣除项量级：A 组侧滑约 0.03–0.06 米/秒，扣除后约束量只上升 2%–5%，不改变上限的量级。"""
    a_group = {"平地0.4": (0.341, 0.026, 0.027), "平地1.0": (0.948, 0.032, 0.036), "崎岖1.0": (0.841, 0.057, 0.059)}
    for scene, (v_par, v_perp, e_w) in a_group.items():
        old, new = v_par + 0.28, v_par - 0.5 * v_perp - 0.3 * e_w + 0.28
        assert 1.0 < old / new < 1.06, (scene, old / new)


def test_lagrange_floor():
    lam = ec.LagrangeMultiplier(init=0.32, lr=0.05, kp=10.0, integral_floor=0.32)
    lam.set_warmup_value(0.32)
    for _ in range(30):
        lam.update(-0.05)  # 约束长期满足
    assert lam.integral == 0.32 and lam.value == 0.32  # 不泄到保底以下
    for _ in range(10):
        lam.update(0.05)
    assert lam.integral > 0.32 and lam.value > lam.integral  # 超限时照常上升
    # 热身期允许低于保底（预热从 0 起步）
    lam.set_warmup_value(0.0)
    assert lam.value == 0.0


def test_scheduler_v4_narrow_band():
    """复现第三版卡死：速度误差稳定在 1.05–1.07 倍，第三版门槛下一直保持，第四版门槛下持续收紧。"""
    for tighten, backoff, expected in ((1.04, 1.09, "hold"), (1.09, 1.12, "tighten")):
        s = make_scheduler(track_mode="error", err_tighten_ratio=tighten, err_backoff_ratio=backoff, cooldown=0)
        s.observe(terrain=3.2, track=0.17, power=40.0)
        s.observe(terrain=3.2, track=0.17, power=40.0)
        decisions = [decide(s, track=0.17 * r, power=s.limit) for r in (1.05, 1.06, 1.07)]
        assert decisions == [expected] * 3, (tighten, decisions)
    assert decide(s, track=0.17 * 1.13, power=s.limit) == "backoff"


def test_scheduler_backoff_headroom():
    """回退不把上限抬到明显高于实测约束量，避免第三版 1050 回退后 λ 泄掉。"""
    # decide() 默认 track=0.8 是奖励模式的数值，误差模式下必须显式给出速度误差
    s = make_scheduler(track_mode="error", err_tighten_ratio=1.09, err_backoff_ratio=1.12, backoff_headroom=0.02, cooldown=0)
    s.observe(terrain=3.2, track=0.17, power=40.0)
    s.observe(terrain=3.2, track=0.17, power=40.0)
    assert decide(s, track=0.17, power=40.0) == "tighten" and math.isclose(s.limit, 36.0)
    # 实测约束量已降到 35：回退到 min(40, max(36, 35*1.02)) = 36，不回到 40
    assert decide(s, track=0.20, power=35.0) == "backoff" and math.isclose(s.limit, 36.0)
    s2 = make_scheduler(track_mode="error", err_tighten_ratio=1.09, err_backoff_ratio=1.12, backoff_headroom=0.02, cooldown=0)
    s2.observe(terrain=3.2, track=0.17, power=40.0)
    s2.observe(terrain=3.2, track=0.17, power=40.0)
    assert decide(s2, track=0.17, power=40.0) == "tighten"
    # 实测约束量仍在 38：回退到 min(40, 38*1.02) = 38.76
    assert decide(s2, track=0.20, power=38.0) == "backoff" and math.isclose(s2.limit, 38.76)
    # headroom=None 时与第三版相同，直接回到 40
    s3 = make_scheduler(track_mode="error", err_tighten_ratio=1.09, err_backoff_ratio=1.12, cooldown=0)
    s3.observe(terrain=3.2, track=0.17, power=40.0)
    s3.observe(terrain=3.2, track=0.17, power=40.0)
    assert decide(s3, track=0.17, power=40.0) == "tighten"
    assert decide(s3, track=0.20, power=35.0) == "backoff" and math.isclose(s3.limit, 40.0)


def test_scheduler_drift_monitor():
    s = make_scheduler(track_mode="error", err_tighten_ratio=1.09, err_backoff_ratio=1.12,
                       drift_tighten_ratio=1.25, drift_backoff_ratio=1.40, cooldown=0)
    obs = dict(terrain=3.2, track=0.17, power=40.0, lateral=0.04, yaw_error=0.05)
    s.observe(**obs)
    s.observe(**obs)
    assert math.isclose(s.lateral_ref, 0.04) and math.isclose(s.yaw_error_ref, 0.05)

    def decide_drift(**kw):
        values = dict(obs, power=s.limit, **kw)
        assert s.observe(**values) == "hold"
        return s.observe(**values)

    assert decide_drift() == "tighten"
    assert decide_drift(lateral=0.054) == "hold"  # 1.35 倍：不收紧也不回退
    assert decide_drift(lateral=0.066) == "backoff"  # 1.65 倍（第三版 1049 评估中的侧滑倍数）：回退
    assert decide_drift(yaw_error=0.075) == "hold"  # 偏航误差 1.5 倍：已回退过，只保持
    restored = make_scheduler(track_mode="error", err_tighten_ratio=1.09, err_backoff_ratio=1.12,
                              drift_tighten_ratio=1.25, drift_backoff_ratio=1.40)
    restored.load_state_dict(s.state_dict())
    assert restored.state_dict() == s.state_dict()
    try:
        s.observe(terrain=3.2, track=0.17, power=40.0)  # 开启监控却不给漂移量
    except ValueError:
        pass
    else:
        raise AssertionError("未拒绝缺少漂移量的输入")


def test_invalid_inputs():
    for bad in (dict(limit_init=10.0), dict(shrink=1.0), dict(limit_init=None, warmup=0),
                dict(terrain_floor=0.0), dict(cooldown=-1), dict(track_mode="speed"),
                dict(track_mode="error", warmup=0), dict(err_tighten_ratio=1.1, err_backoff_ratio=1.05),
                dict(drift_tighten_ratio=1.25), dict(drift_tighten_ratio=1.4, drift_backoff_ratio=1.25),
                dict(backoff_headroom=-0.1)):
        try:
            make_scheduler(**bad)
        except ValueError:
            continue
        raise AssertionError(f"未拒绝非法参数 {bad}")
    try:
        make_scheduler().observe(terrain=float("nan"), track=0.8, power=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("未拒绝 NaN 输入")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"通过 {test.__name__}")
    print(f"全部 {len(tests)} 项通过")
    sys.exit(0)
