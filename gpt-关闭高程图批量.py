"""两种高程条件在相同四个场景配对运行，保存独立输出。"""
from pathlib import Path
import concurrent.futures
import json
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'gpt-复现记录/gpt-关闭高程图对照-20261022'
OUT.mkdir(parents=True,exist_ok=True)
def worker(condition,gpu):
    for scene in ('平地1.0','平地1.6','崎岖地形','8.5厘米楼梯'):
        run=OUT/condition/scene
        if (run/'gpt-评估完成.json').exists():continue
        print('开始',condition,scene,flush=True);start=time.monotonic()
        with (OUT/f'gpt-{condition}-{scene}.log').open('w') as f:
            r=subprocess.run([sys.executable,str(ROOT/'gpt-关闭高程图评估.py'),'--condition',condition,'--scene',scene,'--gpu',str(gpu)],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
        status={'条件':condition,'场景':scene,'退出码':r.returncode,'耗时秒':time.monotonic()-start}
        (OUT/f'gpt-{condition}-{scene}-进程状态.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(status,ensure_ascii=False),flush=True)
        if r.returncode or not (run/'gpt-评估完成.json').exists():raise RuntimeError(f'运行失败：{condition}/{scene}，不能计为机器人失败')
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    fs=[pool.submit(worker,'开启高程图',4),pool.submit(worker,'关闭高程图',5)]
    for f in concurrent.futures.as_completed(fs):f.result()
print('八组评估全部完成',flush=True)
