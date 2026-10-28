"""从训练事件记录绘制迁移前后趋势，不访问或修改训练进程。"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "gpt-复现记录/gpt-迁移后训练检查-20261011.json"
limit = json.loads(REPORT.read_text())["最新迭代"]
base = ROOT / "locomposition/logs/clean_rl/gpt-go2-default-seed46"
runs = [EventAccumulator(str(base / name), size_guidance={"scalars": 0}).Reload()
        for name in ["2026-10-09-13-45-51", "2026-10-10-09-56-51"]]
font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
plt.rcParams["font.family"] = font.get_name()
plt.rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)

def series(ax, tag, label, color):
    for i, run in enumerate(runs):
        events = [e for e in run.Scalars(tag) if e.step <= (1299 if i == 0 else limit)]
        x = np.array([e.step for e in events])
        y = np.array([e.value for e in events])
        n = min(100, len(y))
        ax.plot(x[n-1:], np.convolve(y, np.ones(n)/n, mode="valid"),
                color=color, label=label if i == 1 else None, linewidth=1.6)

ax = axes[0, 0]
for i, run in enumerate(runs):
    es = [e for e in run.Scalars("Loss/learning_rate") if e.step <= (1299 if i == 0 else limit)]
    x = np.array([e.step for e in es]); t = np.array([e.wall_time for e in es])
    ax.plot(x[100:], (t[100:]-t[:-100])/(x[100:]-x[:-100]), color="#2563eb")
ax.set_title("每轮耗时：最近 100 轮均值")
ax.set_ylabel("秒/轮")
series(axes[0, 1], "Curriculum/terrain_levels", "地形等级", "#2563eb")
axes[0, 1].set_title("平均地形等级")
series(axes[1, 0], "Metrics/base_velocity/error_vel_xy", "平面速度误差", "#2563eb")
series(axes[1, 0], "Metrics/base_velocity/error_vel_yaw", "偏航角速度误差", "#ea580c")
axes[1, 0].set_title("训练日志跟踪误差（非论文均方根误差）")
axes[1, 0].legend()
series(axes[1, 1], "Episode_Reward/track_lin_vel_xy_exp", "平面速度奖励", "#2563eb")
series(axes[1, 1], "Episode_Reward/track_ang_vel_z_exp", "偏航角速度奖励", "#ea580c")
axes[1, 1].set_title("速度跟踪奖励")
axes[1, 1].legend()
series(axes[2, 0], "Curriculum/power/weight", "能耗惩罚权重", "#2563eb")
axes[2, 0].axhline(.008, color="#64748b", linestyle=":", label="最终权重 0.008")
axes[2, 0].set_title("能耗课程进度")
axes[2, 0].legend()
for term, label, color in [("base_orientation", "姿态", "#2563eb"),
                           ("joint_acceleration", "关节加速度", "#ea580c"),
                           ("joint_torque", "关节力矩", "#16a34a")]:
    series(axes[2, 1], "Episode_Constraint_violation/"+term, label, color)
axes[2, 1].set_title("训练回合约束违规率（%）")
axes[2, 1].legend()
for ax in axes.flat:
    ax.axvline(1300, color="#64748b", linestyle="--", alpha=.7)
    ax.set_xlabel("训练迭代")
    ax.grid(alpha=.2)
    ax.set_xlim(0, limit)
fig.suptitle(f"GPU2（图形处理器 2）迁移后训练检查 · 截至第 {limit} 轮\n"
             "虚线：第 1300 轮开始迁移续训；指标为 100 轮移动均值，迁移前后分别平滑", fontsize=15)
out = ROOT / "gpt-复现记录/gpt-迁移后训练趋势-20261011.png"
fig.savefig(out, dpi=160)
print(out)
