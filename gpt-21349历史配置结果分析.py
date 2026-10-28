"""汇总作者21349历史配置对照；保留原始结果与证据边界。"""
from pathlib import Path
import csv
import difflib
import hashlib
import importlib.util
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "gpt-复现记录/gpt-21349历史配置核查-20261019"
PREVIOUS = ROOT / "gpt-复现记录/gpt-作者21349评估-20261019"
spec = importlib.util.spec_from_file_location("previous_analysis", ROOT / "gpt-作者21349结果分析.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def main():
    results = {}
    for label, parent in (("当前默认配置", PREVIOUS / "默认完整评估"),
                          ("仅取消质心偏移", OUT / "仅取消质心偏移"),
                          ("历史公开评估配置", OUT / "历史公开评估配置")):
        paths = list(parent.glob("eval_*/metrics_summary.json"))
        if len(paths) != 1:
            raise RuntimeError(f"{label}尚未有唯一完整结果：{paths}")
        results[label], arrays = analysis.analyze(paths[0].parent)
        results[label]["评估器能耗分母标称质量千克"] = results[label].pop("仿真质量千克")
        arrays.close()
    comparisons = []
    for label, result in results.items():
        comparisons.append({"条件": label, "主区间运输能耗": result["主区间运输能耗"],
                            "作者口径速度均方根误差米每秒": result["全窗口"]["作者两轴RMSE平均米每秒"],
                            "五类最大软约束违规率百分比": result["全窗口"]["论文五类最大违规率百分比"],
                            "固定场景完成数": result["固定场景完成数"],
                            "评估器能耗分母标称质量千克": result["评估器能耗分母标称质量千克"]})
        with (OUT / f"gpt-{label}-速度扫描.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result["速度扫描"][0])
            writer.writeheader()
            writer.writerows(result["速度扫描"])
    with (OUT / "gpt-环境差异对照.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=comparisons[0])
        writer.writeheader()
        writer.writerows(comparisons)

    # 文本只应有三轴随机质心范围这一处配置差异。
    current_cfg = (OUT / "冒烟-当前配置核对/gpt-构造环境配置.yaml").read_text()
    com_cfg = (OUT / "仅取消质心偏移/gpt-构造环境配置.yaml").read_text()
    previous_snapshot = next((PREVIOUS / "默认完整评估").glob("eval_*/gpt-当前评估配置/env.yaml"))
    smoke_snapshot = next((OUT / "冒烟-当前配置核对").glob("eval_*/gpt-当前评估配置/env.yaml"))
    assert previous_snapshot.read_bytes() == smoke_snapshot.read_bytes()
    old_range = """com_range:
        x: !!python/tuple
        - -0.03
        - 0.03
        y: !!python/tuple
        - -0.03
        - 0.03
        z: !!python/tuple
        - -0.03
        - 0.03"""
    assert current_cfg.count(old_range) == 1
    assert current_cfg.replace(old_range, old_range.replace("-0.03", "0.0").replace("0.03", "0.0")) == com_cfg
    (OUT / "gpt-仅质心对照实际配置差异.patch").write_text("".join(difflib.unified_diff(
        current_cfg.splitlines(True), com_cfg.splitlines(True), fromfile="当前默认配置", tofile="仅取消质心偏移")))
    runtime = {}
    for label, folder in (("当前默认配置", "冒烟-当前配置核对"),
                          ("仅取消质心偏移", "仅取消质心偏移"), ("历史公开评估配置", "历史公开评估配置")):
        raw = json.loads((OUT / folder / "gpt-运行时质量与质心.json").read_text())
        idx = raw["刚体名称"].index("base")
        runtime[label] = {"机身质量千克": raw["各刚体质量千克"][idx],
                          "总质量千克": sum(raw["各刚体质量千克"]),
                          "机身局部质心米": raw["各刚体局部质心位置与旋转"][idx][:3]}
    assert runtime["当前默认配置"]["总质量千克"] == runtime["仅取消质心偏移"]["总质量千克"]
    for label in results:
        result = results[label]
        result["按实际质量修正的运输能耗参考"] = result["主区间运输能耗"] * result["评估器能耗分母标称质量千克"] / runtime[label]["总质量千克"]
    base, com, historical = comparisons
    effects = {key: {"原值": base[key], "仅取消质心偏移": com[key],
                    "绝对变化": com[key] - base[key], "相对降低百分比": 100 * (1 - com[key] / base[key])}
               for key in ("主区间运输能耗", "作者口径速度均方根误差米每秒", "五类最大软约束违规率百分比")}
    limits = [
        "21349原始单次运行env.yaml、agent.yaml及训练日志未找到；训练日期来自文件名，不能据此证明精确代码提交、训练种子或实际参数。",
        "历史公开配置依据2d1daae附近源码，与首次公开模型时环境源码的功能配置一致；它是可核验的候选配置，不是恢复出的原运行快照。",
        "仅取消质心偏移组保留事件与随机采样次数，实际质量相同；配置快照唯一差异为三轴偏移分布由±0.03米变为零。结果只支持此模型、此种子、此协议下的条件性影响。",
        "历史配置组同时修改多个因素，随机数流也随事件删除改变；不得把组合结果全部归因于某个单独参数。",
        "历史配置组仍使用当前Isaac Sim 5.1、Isaac Lab、机器人资产、地形实现和统一评估流程，没有重建4.5时期完整软件栈。",
        "历史sim.max_position_iteration_count等顶层属性按源码保留，但并非已验证会传入PhysX（物理引擎）的求解器设置，不能称为已复原有效求解器配置。",
        "全部为同一导出模型、评估种子46，各14500步；论文是12个训练种子的均值与95%置信区间，未做统计等价检验。",
        "运输能耗按0.6—1.6米/秒六点等权平均；速度误差沿用作者两轴均方根误差平均；软约束从轨迹补算五类最大单分量违规率，姿态阈值为总倾角0.1弧度。论文完整统计窗口未被完全确认。",
        "场景未完成表示窗口内提前重置，不能自动判为摔倒；所有窗口都保留，不通过丢弃失败场景改善指标。",
        "原评估器用robot.data.default_mass的标称15.019千克计算运输能耗，并非随机化后的实际质量；主表沿用其公式，另保存实际质量修正参考。当前默认组实际质量来自同配置同种子的100步构造核对。",
        "后期历史分析文档把LEP（论文主方法标签）关联env_id_97，属于实验组织线索，并非21349与论文表格的官方逐模型对应清单。论文实机或视频是否使用21349仍未知。",
        "检索覆盖所获取9个远端分支的1248条历史提交及相关历史文件，关键词检索对757份本地可读文本对象进行；并未声称所有缺失二进制或远程私有日志都已审计。",
    ]
    payload = {"比较": comparisons, "条件结果": results, "运行时质量与质心": runtime,
               "仅取消质心偏移的变化": effects, "实际配置唯一差异核验通过": True,
               "论文参考": {"运输能耗": .28, "速度均方根误差米每秒": .19, "最大软约束违规率百分比": .50},
               "限制": limits}
    (OUT / "gpt-历史配置对照结果.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    plt.rcParams.update({"axes.unicode_minus": False, "font.family": font.get_name()})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for label, result in results.items():
        rows = result["速度扫描"]
        x = [r["指令速度米每秒"] for r in rows]
        axes[0, 0].plot(x, [r["运输能耗"] for r in rows], "o-", label=label)
        axes[0, 1].plot(x, [r["平均实际前向速度米每秒"] for r in rows], "o-", label=label)
    axes[0, 0].axvspan(.6, 1.6, alpha=.07, color="green")
    axes[0, 0].set(title="分速度运输能耗", xlabel="指令速度（米/秒）", ylabel="运输能耗")
    axes[0, 1].plot([.2, 2], [.2, 2], "k--", label="理想跟踪")
    axes[0, 1].set(title="前向速度跟踪", xlabel="指令速度（米/秒）", ylabel="平均实际速度（米/秒）")
    for ax in axes[0]:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    names = list(results["当前默认配置"]["全窗口"]["五类软约束最大单分量违规率百分比"])
    for j, (label, result) in enumerate(results.items()):
        axes[1, 0].bar(np.arange(5) + (j - 1) * .25,
                       list(result["全窗口"]["五类软约束最大单分量违规率百分比"].values()), .25, label=label)
    axes[1, 0].axhline(.5, color="red", linestyle="--", label="论文最大值中心0.50%")
    axes[1, 0].set(xticks=np.arange(5), xticklabels=names, ylabel="违规时间比例（%）", title="五类软约束，全评估窗口")
    axes[1, 0].legend(fontsize=8)
    xs = np.arange(3)
    axes[1, 1].bar(xs, [c["作者口径速度均方根误差米每秒"] for c in comparisons])
    axes[1, 1].axhline(.19, color="red", linestyle="--", label="论文中心0.19米/秒")
    axes[1, 1].set(xticks=xs, xticklabels=list(results), ylabel="米/秒", title="作者口径两轴速度均方根误差平均")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("作者21349环境配置诊断：同模型、种子46；历史组仅恢复可移植配置", fontsize=14)
    fig.savefig(OUT / "gpt-历史配置与质心对照.png", dpi=170)
    plt.close(fig)

    lines = ["作者21349模型：仓库历史、原始配置线索与环境不匹配核查", "",
             "本次已完成历史追溯、运行时配置核对及两组新增完整仿真。原始单次训练快照仍未找到，不能宣称恢复了21349的精确训练环境或复现论文多种子结果。", "",
             "一、模型身份与论文关系", "",
             "模型文件：2025-06-28-17-13-04_21349_traced_deterministic.pt。",
             "SHA256（文件校验摘要）：ac81a9195f4c55461c8bcb4b6ccfc61f7d6cdc509176f64421110875d7efcfac。",
             "完整Git（版本控制系统）历史显示首次加入为2025-07-08提交fb5cd528；首次加入与当前文件对象均为c1a77326d56be9df6a9f9ee5ff4c6baf449d53e9，文件未变。浅克隆显示的最近起点不能当首次提交。",
             "作者在2026-01-28提交a3a1e490明确将它标为env75、带高程图的标准策略；2026-04-11提交a02e4a7c将同次训练6049标为能耗最小化前的模型。21349应输入真实高程图，不能误用env79/env80的零高程图。",
             "后期历史分析文档将LEP关联env_id_97_regression_perf_shorter_push_disturbance_seeding、LP（无能耗最小化对照标签）关联env_id_102。证据位于历史文档14第447—452行，来源提交651924cd2747e75c8a6c9150b58cea0f8af6a56d。这支持它们是不同命名实验组，但不足以确定21349是否出现在论文实机/视频中。", "",
             "二、训练日期附近可恢复的公开配置", "",
             "候选提交2d1daae位于2025-06-28，不能证明就是文件名17:13那次训练的精确提交。其环境文件与首次公开模型时文件仅注释有差异。未发现随模型提供的params/env.yaml、params/agent.yaml、原始训练检查点、优化器状态或训练日志。",
             "输入保持236维：角速度3、指令3、重力3、关节角历史36、关节速度历史36、上一动作12、13×11高程图143；三帧历史从新到旧，关节角为绝对角度，模型内部归一化。动作缩放0.8、控制50赫兹、比例/微分增益25/0.5与当前接入一致。",
             "当时公开优化器配置为每轮24步、30000轮；环境7500并行。能耗课程采用deprl自适应方案，7月24日才默认改为线性课程，9月23日才改为从第0步增加能耗惩罚。这些是源码默认值，不等于已确认的21349实际启动参数。",
             "2025-08-19提交4745ee22修复CaT（约束终止机制）训练中的重复奖励缩放；21349在修复前已公开。因此不能仅凭后来源码解释旧模型训练行为，也不能断言它实际训练必然包含该问题或问题导致了多少性能差距。",
             "当时安装脚本固定Python 3.10、PyTorch（张量计算框架）2.5.1+cu121、Isaac Sim（机器人仿真器）4.5.0、Isaac Lab（机器人学习框架）9be0de5f6196374fa793dc33978434542e7a40d3。当前使用Python 3.11、PyTorch 2.7、Isaac Sim 5.1；本次没有重装旧软件栈。", "",
             "三、已确认的环境变化", "",
             "1. 2026-04-30新增机身质心各轴±0.03米和机身质量±1.5千克随机化，当前评估继承它们；旧模型年代公开配置没有。",
             "2. 当前评估开启观测噪声、新增射线横向漂移和高度偏移；历史评估关闭观测噪声。",
             "3. 当前评估重置关节姿态为髋0、大腿1.1、小腿-1.8；历史在默认站姿附近±5%随机初始化。重置偏航范围也由±1.57变为±π。",
             "4. 历史评估电机为力矩限幅23.5、堵转力矩23.5、速度上限30、摩擦0；当前为23.5/20/20/0.05。此前换训练电机70/70/30/0未解决差距，不应把训练电机直接当历史评估电机。",
             "5. 当前仿真和历史仿真的高程图维度与排列一致。实机部署网格比仿真向前移动2厘米，是另一个边界差异，本次保留历史仿真定义。", "",
             "四、完整对照结果", ""]
    for c in comparisons:
        lines.append(f"{c['条件']}：运输能耗{c['主区间运输能耗']:.6f}；速度均方根误差{c['作者口径速度均方根误差米每秒']:.6f}米/秒；五类最大软约束违规率{c['五类最大软约束违规率百分比']:.4f}%；完整固定场景{c['固定场景完成数']}/21。")
    lines += ["论文主表参考：运输能耗0.28±0.01；速度均方根误差0.19±0.01米/秒；最大软约束违规率0.50±0.08%。", ""]
    for key, change in effects.items():
        lines.append(f"仅取消质心随机偏移，{key}由{change['原值']:.6f}变为{change['仅取消质心偏移']:.6f}，相对降低{change['相对降低百分比']:.2f}%。")
    lines += ["", "运行时实际机身质心/质量："]
    for label, values in runtime.items():
        lines.append(f"{label}：{json.dumps(values, ensure_ascii=False)}")
    lines += ["取消偏移保留资产原生质心，不是将质心位置强行设为原点。默认构造核对与仅取消质心组总质量完全一致，已保存实际配置差异补丁。",
             "另发现评估器eval.py第1920行使用default_mass（标称质量）计算运输能耗：三组分母均为15.019千克，而当前默认构造核对与仅取消质心组实际总质量约15.322千克，历史组约15.019千克。为保持作者口径，主表不改分母；按实际质量修正的参考值另存结果JSON（结构化数据文件），不能将约2%的分母差异解释为全部能耗差距。", "",
             "解读：单独取消后来加入的质心随机偏移，使本次最大违规率降低80.90%、速度误差降低25.73%、运输能耗降低10.38%，支持环境不匹配是当前评估差的重要来源。历史组合配置进一步改善，但无法分离噪声、姿态、电机等因素各自的贡献。",
             "历史组速度误差低于论文中心值，但能耗仍比0.28高约16.0%，姿态主导的最大违规率仍约为0.50%的4.06倍；未达到论文各指标的整体表现。它在随机指令窗口第2322步仍有一次非固定窗口末尾的自动重置，21/21仅指固定场景。",
             "配置改善不等于证明21349就是论文主表模型；不能把一个种子的变化推广为全部种子的统计结论。", "",
             "五、证据时间线（均可由本地完整历史仓库复查）"]
    for entry in sorted(json.loads((OUT / "gpt-关键时间线.json").read_text()), key=lambda x: x["原提交时区时间"]):
        lines.append(f"{entry['原提交时区时间']} {entry['事件']}：{entry['链接']}")
    lines += ["", "六、限制"] + [f"{i+1}. {limit}" for i, limit in enumerate(limits)]
    lines += ["", "七、文件与复算", "历史与单因素仿真入口：根目录gpt-21349历史配置对照.py；已有结果时拒绝覆盖。",
             "数据复算：.venv-v100/bin/python gpt-21349历史配置结果分析.py。",
             "原始轨迹和指标位于每个条件的eval_checkpoint_deterministic_seed_46_action_delay_0目录；运行配置、质量与质心、模型接口审核均保留。",
             "当前默认基线来自上一轮完整评估；新增100步默认核对只用于记录初始运行时质心与质量，不进入本报告性能指标。", "",
             "下一步：若目标是21349的精确复现，需要获得env75原运行配置与日志，再对齐4.5时期仿真器和资产；若目标是论文主表，应优先确定env_id_97对应的检查点与12个种子记录。当前不启动重新训练。"]
    (OUT / "gpt-21349历史与环境不匹配核查.txt").write_text("\n".join(lines) + "\n")
    artifacts = [p for p in OUT.rglob("*") if p.is_file() and p.suffix in (".txt", ".patch", ".json", ".yaml", ".csv", ".npz") and p.name != "gpt-证据文件校验.json"]
    (OUT / "gpt-证据文件校验.json").write_text(json.dumps({str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(artifacts)}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(comparisons, ensure_ascii=False, indent=2))
    print(json.dumps(effects, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
