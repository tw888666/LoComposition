"""读取本项目的本地训练标量，不需要启动仿真器。"""
from pathlib import Path
import argparse
import json
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=46)
args = parser.parse_args()
root = Path(__file__).resolve().parent
run_root = root / "locomposition/logs/clean_rl" / f"gpt-go2-default-seed{args.seed}"
runs = sorted(p for p in run_root.glob("*") if p.is_dir())
if not runs:
    raise SystemExit("尚未找到该种子的训练目录。")
run = runs[-1]
acc = EventAccumulator(str(run), size_guidance={"scalars": 0})
acc.Reload()
tags = acc.Tags()["scalars"]
values = {}
for tag in tags:
    if tag.startswith(("Loss/", "Episode_Reward/", "Curriculum/", "Metrics/base_velocity/")):
        points = acc.Scalars(tag)
        if points:
            values[tag] = {"迭代": points[-1].step, "数值": points[-1].value}
models = sorted(run.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
print(json.dumps({"训练目录": str(run), "最新模型": str(models[-1]) if models else None,
                  "最新标量": values}, ensure_ascii=False, indent=2))
