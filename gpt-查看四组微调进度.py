"""读取真实训练日志、模型和阶段文件，显示四组微调进度。"""
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'gpt-四组微调实验-20261026'
now = datetime.now(ZoneInfo('Asia/Shanghai'))
print('北京时间：', now.isoformat(timespec='seconds'))
for label in 'ABCD':
    path = OUT / f'gpt-{label}-状态.json'
    if not path.exists():
        print(label, '尚未启动')
        continue
    state = json.loads(path.read_text())
    text = Path(state['日志']).read_text(errors='replace')
    rows = re.findall(r'\[训练\] 迭代=(\d+)/(\d+) 耗时=([\d.]+)s', text)
    dirs = re.findall(r'\[INFO\] Reserved run directory: (.+)', text)
    if rows:
        completed, total, _ = rows[-1]
        seconds = sum(float(x[2]) for x in rows[-10:]) / len(rows[-10:])
        hours = (int(total)-int(completed))*seconds/3600
        eta = now + timedelta(hours=hours)
        print(f'{label} | {state["阶段"]} | GPU {state["物理GPU"]} | {completed}/{total} | 近期{seconds:.2f}秒/轮 | 估计剩余{hours:.1f}小时 | 预计{eta:%m-%d %H:%M}')
    else:
        print(f'{label} | {state["阶段"]} | GPU {state["物理GPU"]} | 仿真初始化中')
    if dirs:
        models = list(Path(dirs[-1]).glob('model_*.pt'))
        latest = max(models, key=lambda p: int(p.stem.split('_')[1])) if models else None
        print('  最新保存模型：', latest if latest else '尚未保存')
for label in ('A0', 'A', 'B', 'C', 'D'):
    path = OUT / f'gpt-{label}-评估状态.json'
    if path.exists():
        s = json.loads(path.read_text())
        print(f'{label}评估：{s["阶段"]}，{s["完成场景数"]}/{s["总场景数"]}')
total = OUT / 'gpt-总状态.json'
if total.exists():
    print('总状态：', json.loads(total.read_text())['状态'])
