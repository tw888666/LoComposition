"""离线对齐论文 v1 的指标；不修改原模型、训练参数和原始评估文件。"""
from pathlib import Path
import csv
import json
import hashlib
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "locomposition/logs/clean_rl/gpt-go2-default-seed46/2026-10-10-09-56-51"
EVAL = RUN / "eval_checkpoint_30000_seed_46_action_delay_0"
OUT = ROOT / "gpt-复现记录"
STEM = "gpt-论文指标对齐-20261019"
summary = json.loads((EVAL / "metrics_summary.json").read_text())
data = np.load(EVAL / "plots/sim_data.npz", allow_pickle=False)
n = summary["total_sim_steps"]
dt = float(np.median(np.diff(data["sim_times"])))
mass = float(data["total_robot_mass"])
assert n == 14500 and abs(dt - .02) < 1e-9

# 文件中姿态列为 yaw（偏航）、pitch（俯仰）、roll（横滚），不是通常的反序。
angles = data["base_orientation_array"]
tilt = np.arccos(np.clip(np.cos(angles[:, 1]) * np.cos(angles[:, 2]), -1, 1))
gravity_xy = np.sqrt(np.maximum(0, 1 - np.cos(tilt) ** 2))
error = data["base_linear_velocity_body_array"][:, :2] - data["commanded_velocity_array"][:, :2]
soft = {
    "关节力矩": np.abs(data["joint_torques_array"]) > 20,
    "关节速度": np.abs(data["joint_velocities_array"]) > 25,
    "关节加速度": np.abs(data["joint_accelerations_array"]) > 800,
    "动作变化率": np.abs(data["action_rate_array"]) > 80,
    "机身倾角": (tilt > .1)[:, None],
}
valid = np.ones(n, dtype=bool)
valid[np.array(sorted(set(summary["manual_reset_steps"] + summary["automatic_reset_steps"])), dtype=int)] = False

def stats(mask):
    err = error[mask].astype(np.float64)
    rates = {}
    for name, flags in soft.items():
        v = flags[mask]
        rates[name] = {
            "最大单分量违规率百分比": float(v.mean(axis=0).max() * 100),
            "任一分量违规时间占比百分比": float(v.any(axis=1).mean() * 100),
            "全部分量样本违规率百分比": float(v.mean() * 100),
        }
    return {
        "样本数": int(mask.sum()),
        "二维向量RMSE米每秒": float(np.sqrt(np.mean(np.sum(err**2, axis=1)))),
        "两分量平均RMSE米每秒": float(np.sqrt(np.mean(err**2))),
        "前向RMSE米每秒": float(np.sqrt(np.mean(err[:, 0]**2))),
        "横向RMSE米每秒": float(np.sqrt(np.mean(err[:, 1]**2))),
        "五类软约束": rates,
        "五类最大单分量违规率百分比": max(v["最大单分量违规率百分比"] for v in rates.values()),
        "仓库重力水平投影大于0点1的比例百分比": float((gravity_xy[mask] > .1).mean() * 100),
        "姿态逐轴超限解释的最大单轴违规率百分比": float((np.abs(np.arctan2(np.sin(angles[mask, 1:]), np.cos(angles[mask, 1:]))) > .1).mean(axis=0).max() * 100),
    }

windows = {"全评估窗口": np.ones(n, bool), "随机指令段": np.arange(n) < summary["random_sim_steps"]}
scenarios = []
cot = []
for i, (name, *_) in enumerate(summary["fixed_command_scenarios"]):
    start = summary["random_sim_steps"] + i * summary["fixed_command_sim_steps"]
    stop = start + summary["fixed_command_sim_steps"]
    mask = (np.arange(n) >= start) & (np.arange(n) < stop)
    row = {"场景": name, **stats(mask)}
    scenarios.append(row)
    if name.startswith("cot_sweep_walk_x_flat_terrain_"):
        speed = float(name.rsplit("_", 1)[1])
        energy = float(np.abs(data["power_array"][mask]).sum(dtype=np.float64) * dt)
        raw_energy = float(np.abs(data["raw_power_array"][mask]).sum(dtype=np.float64) * dt)
        distance = float(data["distance_increment_array"][mask].sum())
        value = energy / (mass * 9.81 * distance)
        # 独立重算应与原评估器一致；功率单精度累积会有极小舍入差异。
        np.testing.assert_allclose(value, summary["fixed_command_scenarios_metrics"][name]["cost_of_transport"], rtol=2e-5)
        cot.append({"指令速度米每秒": speed, "模拟时长秒": int(mask.sum()) * dt,
                    "平均实际前向速度米每秒": float(data["base_linear_velocity_body_array"][mask, 0].mean()),
                    "机械能焦耳": energy, "水平路程米": distance, "运输能耗": value,
                    "未修复重置步功率的运输能耗": raw_energy / (mass * 9.81 * distance),
                    "纳入论文主区间": .6 <= speed <= 1.6, "超过训练前向指令上限": speed > 1,
                    "前向速度均方根误差米每秒": row["前向RMSE米每秒"]})

# 保留论文要求的最后 500 个训练记录；不能把相关迭代当成独立种子。
acc = EventAccumulator(str(RUN), size_guidance={"scalars": 0}).Reload()
tags = ["Curriculum/terrain_levels", "Metrics/base_velocity/error_vel_xy",
        "Episode_Reward/track_lin_vel_xy_exp", "Curriculum/power/weight"]
tags += ["Episode_Constraint_violation/" + name for name in
         ["joint_torque", "joint_velocity", "joint_acceleration", "action_rate", "base_orientation"]]
train = {}
for tag in tags:
    points = acc.Scalars(tag)[-500:]
    assert len(points) == 500 and points[-1].step == 30000
    train[tag] = {"起始迭代": points[0].step, "结束迭代": points[-1].step,
                  "记录数": len(points), "均值": float(np.mean([p.value for p in points]))}
train_violation = max(v["均值"] for k, v in train.items() if k.startswith("Episode_Constraint_violation/"))

result = {
    "论文": "https://arxiv.org/html/2606.15896v1", "训练种子数": 1, "种子": 46,
    "论文LEP表2": {"平面速度RMSE": [.19, .01], "地形等级": [3.11, .02], "运输能耗": [.28, .01], "违规率百分比": [.50, .08]},
    "论文误差棒含义": "12个训练种子的均值和95%置信区间；本次单种子不生成跨种子置信区间。",
    "训练末500记录": train, "训练末500记录五类违规率最大均值百分比": train_violation,
    "主区间运输能耗均值": float(np.mean([r["运输能耗"] for r in cot if r["纳入论文主区间"]])),
    "速度扫描": cot, "评估窗口": {name: stats(mask) for name, mask in windows.items()},
    "排除重置步敏感性": {name: stats(mask & valid) for name, mask in windows.items()},
    "分场景": scenarios,
    "定义与限制": [
        "COT按六个速度点分别计算E/(M*g*d)，再对0.6至1.6米每秒作等权算术平均；不是混合场景总能耗比值，也不是先合并六段能量和路程。",
        "沿用原评估器重置步功率修复和传送路程修复，并提供未修复功率敏感性。原始文件不改写。",
        "姿态采用总倾角acos(cos(pitch)*cos(roll))>0.1弧度；论文没有更细地指定倾角是合成角还是逐轴角，额外保留仓库重力投影口径。",
        "最大违规率在五类软约束的单分量时间违规率中取最大；同时提供任一分量违规时间占比和分量平均值。论文未完整指定关节维度聚合及评估窗口，不能保证与表2完全同口径。",
        "论文明确训练统计先平均末500记录，但未给出训练RMSE的原始二阶误差；本仓库error_vel_xy是误差向量模长累计值，不能从其均值还原真正RMSE。",
        "RMSE同时提供sqrt(mean(ex^2+ey^2))与sqrt(mean([ex^2,ey^2]))，两者相差sqrt(2)。论文未给出该指标公式及完整采样窗口；两种值都不可凭接近程度选用。",
        "现有自动评估包含4000步随机指令与21个固定场景；不能与训练末500轮统计互换。",
        "当前训练前向指令上限1.0米每秒，论文1.6；PD增益、默认站姿、学习率调度及课程爬升等也有差异。",
        "单种子且迁移为近似续训，无配对基线；不能验证论文56%节能或96%违规改善，也不能声称统计显著优于论文。",
    ],
    "输入文件": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in [EVAL / "metrics_summary.json", EVAL / "plots/sim_data.npz", RUN / "params/env.yaml", ROOT / "gpt-复现记录/gpt-LoComposition论文原文.pdf"]},
}
(OUT / (STEM + ".json")).write_text(json.dumps(result, ensure_ascii=False, indent=2))
with (OUT / (STEM + "-速度扫描.csv")).open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=cot[0].keys()); writer.writeheader(); writer.writerows(cot)
print(json.dumps({"主区间COT": result["主区间运输能耗均值"], "训练末500": train,
                  "全程": result["评估窗口"]["全评估窗口"], "随机": result["评估窗口"]["随机指令段"]}, ensure_ascii=False, indent=2))
