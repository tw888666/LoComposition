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
    assert s.limit == 25.0  # 热身末用实测功率标定
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


def test_invalid_inputs():
    for bad in (dict(limit_init=10.0), dict(shrink=1.0), dict(limit_init=None, warmup=0),
                dict(terrain_floor=0.0), dict(cooldown=-1)):
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
