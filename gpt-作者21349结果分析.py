"""分析作者21349完整仿真，保留作者指标并补算论文五类软约束。"""
from pathlib import Path
import argparse
import csv
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / "gpt-复现记录/gpt-作者21349评估-20261019"
BASELINE = ROOT / "locomposition/logs/clean_rl/gpt-go2-default-seed46/2026-10-10-09-56-51/eval_checkpoint_30000_seed_46_action_delay_0"


def analyze(directory):
    path = directory / "metrics_summary.json"
    summary = json.loads(path.read_text())
    data = np.load(directory / "plots/sim_data.npz", allow_pickle=False)
    n = summary["total_sim_steps"]
    assert n == 14500 and summary["random_sim_steps"] == 4000 and summary["fixed_command_sim_steps"] == 500
    assert summary["action_delay_steps"] == 0 and summary["seed"] == 46
    for key in ("base_linear_velocity_body_array", "joint_torques_array", "joint_velocities_array", "joint_accelerations_array", "action_rate_array", "base_orientation_array"):
        assert len(data[key]) == n and np.isfinite(data[key]).all(), key
    angles = data["base_orientation_array"]
    tilt = np.arccos(np.clip(np.cos(angles[:, 1]) * np.cos(angles[:, 2]), -1, 1))
    soft = {
        "关节力矩": np.abs(data["joint_torques_array"]) > 20,
        "关节速度": np.abs(data["joint_velocities_array"]) > 25,
        "关节加速度": np.abs(data["joint_accelerations_array"]) > 800,
        "动作变化率": np.abs(data["action_rate_array"]) > 80,
        "总倾角": (tilt > .1)[:, None],
    }
    def stats(mask):
        error = (data["base_linear_velocity_body_array"][mask, :2] - data["commanded_velocity_array"][mask, :2]).astype(float)
        rates = {key: float(value[mask].mean(0).max() * 100) for key, value in soft.items()}
        return {
            "作者两轴RMSE平均米每秒": float(np.sqrt(np.mean(error ** 2, axis=0)).mean()),
            "二维向量RMSE米每秒": float(np.sqrt(np.mean(np.sum(error ** 2, axis=1)))),
            "五类软约束最大单分量违规率百分比": rates,
            "论文五类最大违规率百分比": max(rates.values()),
            "重力水平投影违规率百分比": float((np.sin(tilt[mask]) > .1).mean() * 100),
        }
    all_mask = np.ones(n, dtype=bool)
    full = stats(all_mask)
    author_rmse = (summary["base_linear_velocity_x_rms_error"] + summary["base_linear_velocity_y_rms_error"]) / 2
    np.testing.assert_allclose(full["作者两轴RMSE平均米每秒"], author_rmse, rtol=2e-5)
    valid = all_mask.copy()
    valid[np.asarray(sorted(set(summary["manual_reset_steps"] + summary["automatic_reset_steps"])), dtype=int)] = False
    dt = float(np.median(np.diff(data["sim_times"])))
    assert abs(dt - .02) < 1e-8
    rows, scenarios = [], []
    for i, (name, *_rest) in enumerate(summary["fixed_command_scenarios"]):
        start, end = 4000 + i * 500, 4000 + (i + 1) * 500
        mask = (np.arange(n) >= start) & (np.arange(n) < end)
        metric = summary["fixed_command_scenarios_metrics"][name]
        scenario = {"场景": name, "完成且无提前重置": summary["fixed_command_scenario_completion"][name],
                    "平均前向速度米每秒": float(data["base_linear_velocity_body_array"][mask, 0].mean()), **stats(mask)}
        scenarios.append(scenario)
        if name.startswith("cot_sweep_"):
            speed = float(name.rsplit("_", 1)[-1])
            cot = float(np.abs(data["power_array"][mask]).sum(dtype=float) * dt / (float(data["total_robot_mass"]) * 9.81 * data["distance_increment_array"][mask].sum()))
            np.testing.assert_allclose(cot, metric["cost_of_transport"], rtol=2e-5)
            rows.append({"指令速度米每秒": speed, "平均实际前向速度米每秒": scenario["平均前向速度米每秒"],
                         "运输能耗": cot, "前向RMSE米每秒": metric["base_linear_velocity_x_rms_error"],
                         "论文主速度区间": .6 <= speed <= 1.6, "完整场景": scenario["完成且无提前重置"]})
    core = [r["运输能耗"] for r in rows if r["论文主速度区间"]]
    assert len(core) == 6
    result = {
        "评估目录": str(directory), "总控制步数": n, "控制周期秒": dt,
        "模型路径": summary["used_checkpoint_path"], "仿真质量千克": float(data["total_robot_mass"]),
        "主区间运输能耗": float(np.mean(core)), "全窗口": full,
        "随机指令窗口": stats(np.arange(n) < 4000), "排除重置步": stats(valid),
        "固定场景完成数": sum(summary["fixed_command_scenario_completion"].values()),
        "固定场景总数": len(scenarios), "场景": scenarios, "速度扫描": rows,
        "自动重置步": summary["automatic_reset_steps"],
        "输入SHA256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (path, directory / "plots/sim_data.npz")},
    }
    # 联系图只展示接触模式，不把模式相似度冒充论文的定量指标。
    target = next(i for i, row in enumerate(summary["fixed_command_scenarios"]) if row[0] == "walk_x_flat_terrain_1.0mps")
    start = 4000 + target * 500 + 50
    result["平地步态展示起点"] = start
    return result, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    args = parser.parse_args()
    results, arrays = {}, {}
    for label in ("默认完整评估", "训练电机对照"):
        paths = list((args.root / label).glob("eval_*/metrics_summary.json"))
        if len(paths) != 1:
            raise RuntimeError(f"{label} 必须有且仅有一份完整结果，目前为 {len(paths)}")
        results[label], arrays[label] = analyze(paths[0].parent)
    results["本地自训30000模型"], arrays["本地自训30000模型"] = analyze(BASELINE)
    comparison = []
    for label, result in results.items():
        comparison.append({"模型与条件": label, "平地运输能耗": result["主区间运输能耗"],
                           "作者口径速度RMSE米每秒": result["全窗口"]["作者两轴RMSE平均米每秒"],
                           "五类软约束最大违规率百分比": result["全窗口"]["论文五类最大违规率百分比"],
                           "固定场景完成数": result["固定场景完成数"]})
        with (args.root / f"gpt-{label}-速度扫描.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result["速度扫描"][0]); writer.writeheader(); writer.writerows(result["速度扫描"])
    with (args.root / "gpt-论文指标对照.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=comparison[0]); writer.writeheader(); writer.writerows(comparison)
    limits = [
        "作者21349为部署配置env75模型，未确认属于论文主表LEP的12个训练种子；原训练配置和训练日志未公开随附。",
        "每条件使用同一个模型、同一评估种子46；电机对照不是新增训练种子。不能据此生成论文跨训练种子置信区间。",
        "此前自训模型用于参考比较：场景与种子相同，但普通检查点加载会初始化网络并消耗随机数，导出模型加载路径不同，因此不是逐步噪声完全配对的因果实验。",
        "主条件保留仓库默认评估电机：力矩限幅23.5、堵转力矩20、速度限幅20、摩擦0.05；训练电机对照为70/70/30/0。两者均为25/0.5增益。",
        "部署配置的关节速度30和力矩46是控制节点安全边界，不能等同于仿真电机力矩速度曲线。",
        "当前仿真网格前后坐标为-0.30至0.66米，部署网格为-0.28至0.68米，均为13×11、0.08米分辨率、X索引变化最快。仿真继续使用作者仿真定义。",
        "主条件启用当前PLAY任务的观测噪声及现有事件；使用真实仿真地形射线高度，不使用部署配置默认的固定-0.3米高度。",
        "运输能耗按0.6至1.6米每秒六个速度点分别计算后等权平均；保留原评估器重置步处理，不剔除差场景。",
        "速度使用作者公开汇总公式(RMSE_x+RMSE_y)/2；论文未明确完整窗口，数值接近不能证明协议完全相同。",
        "软约束按论文五类分别逐关节统计后取最大；姿态采用总倾角大于0.1弧度，同时保存重力投影口径。公开评估器原始约束集合不同，未混作论文指标。",
        "没有21349原始训练日志，不能恢复训练末500轮地形等级；评估地形等级不可替代。无配对基线和真实机器人，不验证56%节能、96%违规改善或实机成功率。",
    ]
    payload = {"论文主方法": {"运输能耗": [.28, .01], "速度RMSE": [.19, .01], "最大软约束违规率百分比": [.50, .08], "训练地形等级": [3.11, .02]},
               "论文误差棒": "12个训练种子的均值±95%置信区间", "比较": comparison, "条件结果": results, "可比性限制": limits}
    (args.root / "gpt-结果与可比性.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    plt.rcParams.update({"axes.unicode_minus": False, "font.family": font.get_name()})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for label, result in results.items():
        rows = result["速度扫描"]
        speed = [r["指令速度米每秒"] for r in rows]
        axes[0, 0].plot(speed, [r["运输能耗"] for r in rows], "o-", label=label)
        axes[0, 1].plot(speed, [r["平均实际前向速度米每秒"] for r in rows], "o-", label=label)
    axes[0, 0].axvspan(.6, 1.6, alpha=.08, color="green", label="论文主区间")
    axes[0, 0].set(title="各速度点运输能耗（论文0.28是区间均值）", xlabel="前向指令（米/秒）", ylabel="运输能耗")
    axes[0, 1].plot([.2, 2], [.2, 2], "k--", label="理想跟踪")
    axes[0, 1].set(title="前向速度跟踪", xlabel="前向指令（米/秒）", ylabel="平均实际速度（米/秒）")
    for ax in axes[0]: ax.legend(fontsize=8); ax.grid(alpha=.2)
    names = list(results["默认完整评估"]["全窗口"]["五类软约束最大单分量违规率百分比"])
    for j, (label, result) in enumerate(results.items()):
        axes[1, 0].bar(np.arange(5) + (j - 1) * .25, list(result["全窗口"]["五类软约束最大单分量违规率百分比"].values()), .25, label=label)
    axes[1, 0].axhline(.5, linestyle="--", color="red", label="论文最大值中心0.50%")
    axes[1, 0].set(xticks=np.arange(5), xticklabels=names, ylabel="违规时间比例（%）", title="五类软约束（全评估窗口）")
    axes[1, 0].legend(fontsize=8)
    result, data = results["默认完整评估"], arrays["默认完整评估"]
    start = result["平地步态展示起点"]
    contact = data["contact_state_array"][start:start+200]
    axes[1, 1].imshow(contact.T, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0, vmax=1, extent=(1, 5, 3.5, -.5))
    axes[1, 1].set(yticks=range(4), yticklabels=["左前", "右前", "左后", "右后"], xlabel="场景内时间（秒）", title="作者21349平地1米/秒接触时序（深色为接触）")
    fig.suptitle("作者21349模型仿真核验：单模型、评估种子46；电机对照不代表原训练配置", fontproperties=font, fontsize=14)
    fig.savefig(args.root / "gpt-作者模型与论文效果对比.png", dpi=170)
    lines = ["作者21349模型仿真评估结果", "", "模型接入与两组完整评估已完成；在本次可核验的配置下，未复现论文主表表现。以下为单模型条件性比较，不是论文多种子统计复现。", ""]
    for row in comparison:
        lines.append(f"{row['模型与条件']}：运输能耗 {row['平地运输能耗']:.6f}；作者口径速度均方根误差 {row['作者口径速度RMSE米每秒']:.6f} 米/秒；五类软约束最大违规率 {row['五类软约束最大违规率百分比']:.4f}%；固定场景完成 {row['固定场景完成数']}/21。")
    lines += ["", "结果解读：", "两种电机配置的能耗和姿态违规均未接近论文；本次切换电机参数不能消除差距，但不能据此排除其他环境差异。",
              "默认配置1.6米/秒指令的实际平均前向速度约1.235米/秒。姿态超限是五类软约束的最大项，高速平地也存在明显超限。",
              "默认配置有两个提前重置场景：崎岖地形快速斜向、崎岖地形中速斜向；训练电机对照仍有快速斜向场景提前重置。提前重置不直接等同于摔倒，需结合终止原因和轨迹进一步诊断。",
              "两组数据均跑满14500个控制步。场景未完成表示500步窗口内发生提前自动重置，不表示程序提前退出；报告保留了这些场景的数据。"]
    lines += ["", "论文主方法参考：运输能耗0.28±0.01；速度均方根误差0.19±0.01米/秒；最大软约束违规率0.50±0.08%；训练地形等级3.11±0.02。", "", "输入与环境核对：", "236维输入=角速度3+速度指令3+重力投影3+关节角历史36+关节速度历史36+上一动作12+高程图143。", "原模型内部归一化；三帧历史从新到旧；关节角为绝对角度；动作=默认角度+0.8×模型输出；控制50赫兹；比例/微分增益25/0.5。", "", "限制与解释："]
    lines += [f"{i+1}. {text}" for i, text in enumerate(limits)]
    lines += ["", "运行入口：gpt-作者21349评估.sh；复算入口：.venv-v100/bin/python gpt-作者21349结果分析.py。", "原始指标和轨迹保留于各条件eval目录；本报告没有改写原始结果。"]
    (args.root / "gpt-模型接入与论文效果核验.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
