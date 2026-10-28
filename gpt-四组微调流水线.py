"""四组等预算微调、统一评估与汇总；正式运行应由tmux启动。"""
import argparse
import concurrent.futures
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "gpt-四组微调实验-20261026"
BASE = ROOT / "locomposition/logs/clean_rl/gpt-go2-default-seed46/2026-10-10-09-56-51"
PYTHON = ROOT / ".venv-v100/bin/python"
REFERENCE = OUT / "gpt-固定参考尺度.json"
GPUS = {"A": 1, "B": 2, "C": 4, "D": 5}
SCENES = ["平地0.4", "平地0.8", "平地1.0", "崎岖1.0", "楼梯8.5厘米", "坡道0.8"]
SEEDS = [46, 47, 48]


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def environment(gpu, label):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="8", OMNI_KIT_ACCEPT_EULA="Y",
               OMNICLIENT_HUB_MODE="disabled", WANDB_MODE="disabled", MPLBACKEND="Agg",
               PYTHONUNBUFFERED="1", ENV_NAME=f"gpt-energy-finetune-{label}-seed46-20261026")
    return env


def train(label, gpu, smoke=False):
    iterations, envs = (3, 64) if smoke else (5000, 7500)
    tag = f"冒烟-{label}" if smoke else label
    log = OUT / f"gpt-{tag}-训练.log"
    status_path = OUT / f"gpt-{tag}-状态.json"
    if log.exists():
        raise FileExistsError(f"已有训练日志，拒绝覆盖或重复启动：{log}")
    cmd = [str(PYTHON), "scripts/clean_rl/train.py",
           "--task=LoComposition-Go2-Rough-Terrain-Joint-State-History-v0", "--seed=46", "--headless",
           f"--num_envs={envs}", f"--num_iterations={iterations}", "--logger=tensorboard", "--device=cuda:0",
           "--finetune", f"--checkpoint={BASE / 'model_30000.pt'}", f"--energy_branch={label}",
           f"--energy_reference={REFERENCE}", "env.scene.ray_caster.debug_vis=false",
           "env.scene.ray_caster_height_constraints.debug_vis=false", "env.commands.base_velocity.debug_vis=false"]
    status = {"实验组": tag, "阶段": "训练中", "开始时间": now(), "物理GPU": gpu,
              "追加迭代": iterations, "环境数量": envs, "启动参数": cmd, "日志": str(log)}
    save(status_path, status)
    print(f"{now()} 开始训练 {tag}，GPU {gpu}", flush=True)
    start = time.monotonic()
    with log.open("w") as stream:
        proc = subprocess.Popen(cmd, cwd=ROOT / "locomposition", env=environment(gpu, tag),
                                stdout=stream, stderr=subprocess.STDOUT)
        status["进程PID"] = proc.pid
        save(status_path, status)
        code = proc.wait()
    text = log.read_text(errors="replace")
    dirs = re.findall(r"\[INFO\] Reserved run directory: (.+)", text)
    run = Path(dirs[-1]) if dirs else None
    status.update(退出码=code, 训练耗时秒=time.monotonic()-start, 训练结束时间=now(),
                  模型目录=str(run) if run else None)
    if code or run is None or not (run / f"model_{iterations}.pt").is_file():
        status["阶段"] = "训练失败"
        save(status_path, status)
        raise RuntimeError(f"{tag}训练失败，请查看{log}")
    status["阶段"] = "冒烟完成" if smoke else "训练完成"
    save(status_path, status)
    print(f"{now()} {tag}训练完成：{run}", flush=True)
    return run


def evaluate(label, run, checkpoint, gpu):
    status_path = OUT / f"gpt-{label}-评估状态.json"
    status = {"实验组": label, "阶段": "评估中", "开始时间": now(), "模型目录": str(run),
              "总场景数": len(SCENES)*len(SEEDS), "完成场景数": 0}
    save(status_path, status)
    for seed in SEEDS:
        for scene in SCENES:
            cmd = [str(PYTHON), str(ROOT / "gpt-四组微调评估.py"), "--run-dir", str(run),
                   "--label", label, "--checkpoint", str(checkpoint), "--gpu", str(gpu),
                   "--seed", str(seed), "--scene", scene]
            log = OUT / f"gpt-{label}-评估-{scene}-种子{seed}.log"
            with log.open("a") as stream:
                code = subprocess.call(cmd, cwd=ROOT, env=environment(gpu, label), stdout=stream,
                                       stderr=subprocess.STDOUT)
            if code:
                status.update(阶段="评估失败", 失败场景=scene, 失败种子=seed, 退出码=code)
                save(status_path, status)
                raise RuntimeError(f"{label}评估进程失败：{log}")
            status["完成场景数"] += 1
            save(status_path, status)
    status.update(阶段="评估完成", 完成时间=now())
    save(status_path, status)


def branch(label, gpu):
    run = train(label, gpu)
    evaluate(label, run, 5000, gpu)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-branch", choices=list(GPUS), default="C")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if not REFERENCE.is_file():
        raise FileNotFoundError(REFERENCE)
    if args.smoke:
        train(args.smoke_branch, 4, True)
        return
    if not os.environ.get("TMUX"):
        raise RuntimeError("正式训练必须在tmux会话内运行。")
    status = {"状态": "执行中", "开始时间": now(), "四组GPU": GPUS,
              "源模型SHA256": hashlib.sha256((BASE / "model_30000.pt").read_bytes()).hexdigest(),
              "单组追加迭代": 5000, "单组环境交互数": 5000*7500*24,
              "评估种子": SEEDS, "评估场景": SCENES, "错误": []}
    save(OUT / "gpt-总状态.json", status)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(branch, label, gpu): label for label, gpu in GPUS.items()}
        futures[pool.submit(evaluate, "A0", BASE, 30000, 1)] = "A0"
        for future in concurrent.futures.as_completed(futures):
            label = futures[future]
            try:
                future.result()
                print(f"{now()} {label}全部完成", flush=True)
            except Exception as exc:
                status["错误"].append({"实验组": label, "错误": str(exc), "时间": now()})
                save(OUT / "gpt-总状态.json", status)
                print(f"{now()} {label}失败：{exc}", flush=True)
    if status["错误"]:
        status.update(状态="存在失败，需处理", 结束时间=now())
        save(OUT / "gpt-总状态.json", status)
        raise RuntimeError("有实验阶段失败，不生成完整实验结论。")
    subprocess.run([str(PYTHON), str(ROOT / "gpt-四组微调结果汇总.py")], cwd=ROOT, check=True)
    video = ROOT / "gpt-四组微调对比视频.py"
    if video.exists():
        subprocess.run([str(PYTHON), str(video)], cwd=ROOT, check=True)
    status.update(状态="全部完成", 完成时间=now())
    save(OUT / "gpt-总状态.json", status)
    print(f"{now()} 四组微调、原模型对照与统一评估全部完成。", flush=True)


if __name__ == "__main__":
    main()
