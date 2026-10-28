"""等待完整评估落盘后汇总并渲染平地、崎岖场景；逐项记录退出状态。"""
from pathlib import Path
import concurrent.futures
import json
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'gpt-复现记录/gpt-30000模型对照评估-20261021'
def run(name):
    with (OUT/f'gpt-{name}-后处理.log').open('w') as f:
        result=subprocess.run([sys.executable,str(ROOT/f'gpt-30000{name}.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    (OUT/f'gpt-{name}-后处理状态.json').write_text(json.dumps({'任务':name,'退出码':result.returncode},ensure_ascii=False,indent=2))
    if result.returncode:raise RuntimeError(f'{name}失败，请检查日志')
    print(name,'完成',flush=True)

if __name__=='__main__':
    deadline=time.monotonic()+3600
    status=OUT/'gpt-完整评估-进程状态.json'
    while not status.exists():
        if time.monotonic()>deadline:raise TimeoutError('等待完整评估超时；未修改运行进程。')
        time.sleep(10)
    assert json.loads(status.read_text())['结果完整']
    run('结果汇总')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run,name) for name in ('行走视频','崎岖楼梯视频')]
        for f in concurrent.futures.as_completed(futures):f.result()
    print('全部后处理完成',flush=True)
