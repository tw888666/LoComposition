"""在原四场景配置下评估2999次权重，开启高程图。"""
from pathlib import Path
import concurrent.futures,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'gpt-复现记录/gpt-2999与30000权重对照-20261022'
OUT.mkdir(parents=True,exist_ok=True)
def worker(gpu,scenes):
    for scene in scenes:
        run=OUT/'2999次/开启高程图'/scene
        if (run/'gpt-评估完成.json').exists():continue
        print('开始',scene,flush=True);start=time.monotonic()
        with (OUT/f'gpt-2999次-{scene}-评估.log').open('w') as f:
            r=subprocess.run([sys.executable,str(ROOT/'gpt-关闭高程图评估.py'),'--condition','开启高程图','--scene',scene,'--gpu',str(gpu),'--checkpoint','2999','--output-root',str(OUT/'2999次')],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
        status={'迭代':2999,'场景':scene,'退出码':r.returncode,'耗时秒':time.monotonic()-start}
        (OUT/f'gpt-2999次-{scene}-进程状态.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n');print(json.dumps(status,ensure_ascii=False),flush=True)
        if r.returncode or not (run/'gpt-评估完成.json').exists():raise RuntimeError(f'仿真运行失败：{scene}；不计为机器人失败。')
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    futures=[pool.submit(worker,4,('平地1.0','崎岖地形')),pool.submit(worker,5,('平地1.6','8.5厘米楼梯'))]
    for f in concurrent.futures.as_completed(futures):f.result()
print('2999次四场景评估全部完成',flush=True)
