"""在三张显卡上执行完整评估及21次台阶评估，保存独立日志，支持跳过完整结果。"""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'gpt-复现记录/gpt-30000模型对照评估-20261021'

def worker(gpu, jobs):
    for height, seed in jobs:
        label = '完整评估' if height is None else f'阶梯{height:g}厘米种子{seed}'
        run = OUT / '完整评估' if height is None else OUT / f'阶梯评估/{height:g}厘米/种子{seed}'
        if list(run.glob('eval_*/metrics_summary.json')):
            print('跳过已有结果', label, flush=True)
            continue
        cmd = [sys.executable, str(ROOT / ('gpt-30000完整对照评估.py' if height is None else 'gpt-30000阶梯评估.py')), '--gpu', str(gpu)]
        if height is not None:
            cmd += ['--height-cm', str(height), '--seed', str(seed)]
        started = time.monotonic()
        print('开始', label, '物理显卡', gpu, flush=True)
        with (OUT / f'gpt-{label}.log').open('w') as f:
            result = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        success = result.returncode == 0 and bool(list(run.glob('eval_*/metrics_summary.json')))
        status = {'任务': label, '显卡': gpu, '退出码': result.returncode, '结果完整': success, '耗时秒': time.monotonic()-started}
        (OUT / f'gpt-{label}-进程状态.json').write_text(json.dumps(status, ensure_ascii=False, indent=2))
        print(json.dumps(status, ensure_ascii=False), flush=True)
        if not success:
            raise RuntimeError(f'评估运行失败，不计为机器人失败：{label}')

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT/'gpt-30000阶梯统计.py'), '--init'], check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(worker, 2, [(None,46),(8.5,40)]),
                   pool.submit(worker, 4, [(15,s) for s in range(40,50)]),
                   pool.submit(worker, 5, [(20,s) for s in range(40,50)])]
        for f in concurrent.futures.as_completed(futures):
            f.result()
    subprocess.run([sys.executable, str(ROOT/'gpt-30000阶梯统计.py')], check=True)
    print('完整评估及全部21次阶梯评估完成', flush=True)
