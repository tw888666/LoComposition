"""顺序执行同一高度的预定种子；日志与已完成结果均独立保留。"""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--height-cm',type=int,required=True);p.add_argument('--gpu',required=True);a=p.parse_args()
for seed in range(41,50):
 run=ROOT/f'gpt-复现记录/gpt-21349高台阶多种子-20261019/{a.height_cm}厘米/种子{seed}'
 if list(run.glob('eval_*/metrics_summary.json')):print('已完成，跳过',a.height_cm,seed,flush=True);continue
 log=ROOT/f'gpt-复现记录/gpt-21349台阶{a.height_cm}厘米种子{seed}.log'
 started=time.monotonic();print('开始',a.height_cm,'厘米，种子',seed,flush=True)
 with log.open('w') as f:
  result=subprocess.run([sys.executable,str(ROOT/'gpt-21349阶梯评估.py'),'--height-cm',str(a.height_cm),'--seed',str(seed),'--gpu',a.gpu],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 if result.returncode or not list(run.glob('eval_*/metrics_summary.json')):raise SystemExit(f'评估失败：{log}')
 print('完成',a.height_cm,seed,'耗时秒',round(time.monotonic()-started,1),flush=True)
print('此高度全部种子完成',flush=True)
