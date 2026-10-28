#!/usr/bin/env python3
"""从原三万次权重的已有评估轨迹复算四组微调共用的固定参考尺度。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "gpt-复现记录/gpt-中间权重收敛评估-20261022/gpt-运行清单.json"
MODEL = ROOT / "locomposition/logs/clean_rl/gpt-go2-default-seed46/2026-10-10-09-56-51/model_30000.pt"
SCENES = ("平地1.0", "崎岖地形", "8.5厘米楼梯")
SEEDS = (40, 42, 44, 46, 48)
WARMUP_SECONDS = 1.0
MIN_FORWARD_SPEED = 0.1
MAX_TILT_RADIANS = np.pi / 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calculate() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    selected = [r for r in manifest if r["迭代"] == 30000 and r["场景"] in SCENES]
    actual_pairs = {(r["种子"], r["场景"]) for r in selected}
    expected_pairs = {(seed, scene) for seed in SEEDS for scene in SCENES}
    if actual_pairs != expected_pairs or len(selected) != len(expected_pairs):
        raise ValueError("原始清单必须恰好包含三个场景乘以五个种子，不允许重复或缺失。")
    model_hash = sha256(MODEL)
    summaries, powers, torques_squared = [], [], []
    for row in sorted(selected, key=lambda r: (SCENES.index(r["场景"]), r["种子"])):
        directory = Path(row["目录"])
        npz_files = list(directory.rglob("sim_data.npz"))
        if len(npz_files) != 1:
            raise ValueError(f"轨迹文件必须唯一：{directory}")
        source = npz_files[0]
        condition_file = directory / "gpt-评估条件.json"
        reset_file = directory / "gpt-重置前真实状态.json"
        condition = json.loads(condition_file.read_text(encoding="utf-8"))
        reset_records = json.loads(reset_file.read_text(encoding="utf-8"))
        if condition["权重SHA256"] != model_hash:
            raise ValueError(f"原模型校验值不一致：{condition_file}")
        if condition["种子"] != row["种子"] or condition["条件"] != "开启高程图":
            raise ValueError(f"评估条件与清单不一致：{condition_file}")
        if any(any(value for name, value in record["终止项"].items() if name != "time_out") for record in reset_records):
            raise ValueError(f"参考轨迹存在非超时终止，需要人工重新选择：{reset_file}")

        with np.load(source, allow_pickle=False) as data:
            time = data["sim_times"].astype(np.float64)
            tau = data["joint_torques_array"].astype(np.float64)
            velocity = data["joint_velocities_array"].astype(np.float64)
            base_velocity = data["base_linear_velocity_body_array"].astype(np.float64)
            orientation = data["base_orientation_array"].astype(np.float64)
            command = data["commanded_velocity_array"].astype(np.float64)
            automatic_resets = data["automatic_reset_times"].astype(np.float64)
            manual_resets = data["manual_reset_times"].astype(np.float64)
        if tau.shape != velocity.shape or tau.shape != (len(time), 12):
            raise ValueError(f"需要十二个关节且力矩、速度逐帧对齐：{source}")
        if not np.allclose(np.diff(time), 0.02, rtol=0, atol=1e-10):
            raise ValueError(f"预期控制周期为零点零二秒：{source}")
        if len(manual_resets) != 1 or not np.isclose(manual_resets[0], 0):
            raise ValueError(f"预期只在零秒进行场景初始化：{source}")
        if len(automatic_resets) != 1 or not np.isclose(automatic_resets[0], time[-1]):
            raise ValueError(f"预期只在最后一帧发生超时重置：{source}")

        finite = (np.isfinite(tau).all(axis=1) & np.isfinite(velocity).all(axis=1)
                  & np.isfinite(base_velocity).all(axis=1) & np.isfinite(orientation).all(axis=1)
                  & np.isfinite(command).all(axis=1))
        stable_window = (time >= WARMUP_SECONDS) & (time < automatic_resets[0])
        in_range_command = ((command[:, 0] >= 0.1) & (command[:, 0] <= 1.0)
                            & np.isclose(command[:, 1], 0) & np.isclose(command[:, 2], 0))
        forward_walking = base_velocity[:, 0] >= MIN_FORWARD_SPEED
        # 原评估程序按偏航、俯仰、横滚顺序记录；不能把偏航误当成倾斜角。
        upright = (np.abs(orientation[:, 1:3]) < MAX_TILT_RADIANS).all(axis=1)
        mask = finite & stable_window & in_range_command & forward_walking & upright
        if not mask.any():
            raise ValueError(f"有效行走帧为空：{source}")
        power = np.abs(tau[mask] * velocity[mask]).sum(axis=1)
        torque_squared = np.square(tau[mask]).sum(axis=1)
        powers.append(power)
        torques_squared.append(torque_squared)
        indices = np.flatnonzero(mask)
        summaries.append({
            "场景": row["场景"], "评估种子": row["种子"], "地形种子": condition.get("地形种子", condition["种子"]),
            "地形种子来源": "条件文件明确记录" if "地形种子" in condition else "原种子四十六评估未另设地形种子，按评估脚本默认使用评估种子",
            "轨迹文件": str(source), "轨迹SHA256": sha256(source),
            "条件文件": str(condition_file), "条件SHA256": sha256(condition_file),
            "重置前状态文件": str(reset_file), "重置前状态SHA256": sha256(reset_file),
            "原始帧数": len(time), "有效帧数": int(mask.sum()),
            "有效采样时长秒": float(mask.sum() * 0.02),
            "有效首末时间秒": [float(time[indices[0]]), float(time[indices[-1]])],
            "有效首末索引_从零开始": [int(indices[0]), int(indices[-1])],
            "有效索引连续": bool(np.all(np.diff(indices) == 1)),
            "排除启动帧数": int((time < WARMUP_SECONDS).sum()),
            "排除重置及其后帧数": int((time >= automatic_resets[0]).sum()),
            "窗口内额外无效帧数": int((stable_window & ~mask).sum()),
            "自动重置时刻秒": automatic_resets.tolist(),
            "自动重置原因": [record["终止项"] for record in reset_records],
            "P_ref_平均绝对机械功率_瓦": float(power.mean()),
            "Q_ref_平均力矩平方和_牛米平方": float(torque_squared.mean()),
            "最低有效前向速度_米每秒": float(base_velocity[mask, 0].min()),
            "最大有效俯仰横滚绝对角_弧度": float(np.abs(orientation[mask, 1:3]).max()),
        })
    power_reference = float(np.concatenate(powers).mean())
    torque_reference = float(np.concatenate(torques_squared).mean())
    if not (power_reference > 0 and torque_reference > 0):
        raise ValueError("参考尺度必须为正值。")
    return {
        "用途": "四组微调共用的固定参考尺度；训练开始前固定，训练期间不更新。",
        "计算日期": "2026-10-26", "日期时区": "Asia/Shanghai（上海时区）",
        "P_ref": power_reference, "Q_ref": torque_reference, "s": power_reference / torque_reference,
        "符号解释": {"P_ref": "有效帧内十二关节绝对机械功率之和的均值，单位瓦", "Q_ref": "有效帧内十二关节力矩平方和的均值，单位牛米平方", "s": "P_ref除以Q_ref，用于将力矩平方项缩放至与机械功率项相同的参考量级，单位瓦每牛米平方"},
        "公式": {"P_ref": "mean_t(sum_j(abs(tau[t,j] * dq[t,j])))", "Q_ref": "mean_t(sum_j(tau[t,j] ** 2))", "s": "P_ref / Q_ref"},
        "计算精度": "输入数组先转换为双精度浮点数，再计算乘法、平方、求和及均值。",
        "权重文件": str(MODEL), "权重SHA256": model_hash, "checkpoint_sha256": model_hash,
        "来源清单": str(MANIFEST), "来源清单SHA256": sha256(MANIFEST),
        "复算脚本": str(Path(__file__).resolve()), "复算脚本SHA256": sha256(Path(__file__).resolve()),
        "选择依据": [
            "采用原三万次训练权重，正常真实高程图，三个代表性地形乘以五个评估种子。",
            "平地、崎岖地形及八点五厘米楼梯覆盖基础行走与地形适应；当前训练台阶高度配置范围为零点零五至零点二三米。",
            "三个场景前向指令均为每秒一米，位于原训练前向指令范围负零点三至一米每秒内。",
            "排除平地每秒一点六米，避免将训练指令范围外的外推场景用于固定参考量。",
            "每轨迹有效帧数相同，合并全部有效帧取均值等价于三个场景及各自五个种子等权平均。",
        ],
        "有效行走掩码": {
            "启动稳定时间秒": WARMUP_SECONDS,
            "时间条件": "sim_times >= 1.0 且 sim_times < 首次自动重置时刻；本数据为一秒至九点九六秒，包含两端。",
            "重置排除原因": "原评估在环境推进后记录力矩和关节速度，环境会在推进内部自动重置；九点九八秒的超时帧已混入重置状态，必须排除。",
            "速度条件": "机身坐标系前向实际速度至少每秒零点一米；前向指令每秒零点一至一米，横向和偏航指令为零。",
            "姿态条件": "俯仰和横滚绝对值均小于四十五度；数组顺序是偏航、俯仰、横滚。",
            "数值条件": "力矩、关节速度、机身速度、姿态和指令均为有限数。",
            "终止核查": "全部十五条轨迹均只在最后一帧超时，无机身接触或翻倒终止；窗口内额外无效帧数为零。",
        },
        "采样说明": "使用每个零点零二秒控制周期末保存的力矩和关节速度快照；不是每个物理子步的积分值。固定尺度仅作为奖励数值归一化参考，不将其解释为电机电池电功率。",
        "训练配置依据": {
            "速度与地形配置": str(ROOT / "locomposition/exts/locomposition/locomposition/tasks/locomotion/velocity/config/solo12/locomposition_go2_rough_terrain_env_cfg.py"),
            "地形范围配置": str(ROOT / "dependencies/IsaacLab/source/isaaclab/isaaclab/terrains/config/rough.py"),
        },
        "轨迹总数": len(summaries), "有效总帧数": sum(r["有效帧数"] for r in summaries),
        "有效总采样时长秒": sum(r["有效采样时长秒"] for r in summaries), "逐轨迹记录": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("gpt-固定参考尺度.json"), help="输出中文参考尺度文件路径")
    args = parser.parse_args()
    result = calculate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"输出": str(args.output.resolve()), "P_ref": result["P_ref"], "Q_ref": result["Q_ref"], "s": result["s"], "有效总帧数": result["有效总帧数"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
