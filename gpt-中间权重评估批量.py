"""八个已有权重、四个固定场景、五个动力学及观测种子的配对评估。"""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,queue,subprocess,sys,threading,time
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'gpt-复现记录/gpt-中间权重收敛评估-20261022'
TRAIN=ROOT/'locomposition/logs/clean_rl/gpt-go2-default-seed46/2026-10-10-09-56-51'
ITERATIONS=(2999,4999,9999,12499,14999,19999,24999,30000)
SEEDS=(40,42,44,46,48)
SCENES=('平地1.0','平地1.6','崎岖地形','8.5厘米楼梯')
def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol={'迭代列表':ITERATIONS,'评估种子':SEEDS,'地形种子':46,'场景':SCENES,'每场秒数':10,'高程图':'开启真实高程图','权重':{},
      '随机化':'固定地形布局，改变动力学及观测噪声；取消质心随机偏移，沿用上一轮其他设置。不是五种地形布局。',
      '平地与崎岖通过定义':'10秒内无非超时终止，首次重置前世界负X净前进至少为指令速度×10秒的70%，末帧侧向偏移不超过1米。',
      '楼梯通过定义':'沿用既定四足全部越过平台边界、机身直立并连续保持10步的判据；只在首次重置前判断。',
      '速度误差定义':'前向速度均方根误差，单位米/秒。另保留前向/侧向两轴均方根误差均值。',
      '机械功定义':'逐时刻十二关节|力矩×角速度|之和积分，除以水平路径长度；单位焦耳/米，不是电池能耗。',
      '主连续指标窗口':'每个场景和种子，以八个权重中最早首次重置前的共同有效窗口计算；不拼接自动重置，不让失败权重独享较短窗口。包含起步。',
      '误差带':'连续指标按五个配对种子整组重采样，每组包含四场景，给出95%自助法区间；五个种子的区间仅供参考。通过率直接报告成功数/20及逐场景成功数/5。',
      '接近30000次参考标准':{'总体通过率至少':.9,'各场景通过率至少':.8,'比30000次总体通过率至多低':.05,'速度误差上限':'30000次×1.05 + 0.01米/秒','每米机械功上限':'30000次×1.10','稳定性':'该点及所有后续采样点均达标，且至少含三个采样点。'},
      '限制':'结论限于这次训练、固定四场景、五个种子；不证明所有地形收敛，不以最后权重天然最佳。'}
    for n in ITERATIONS:
        p=TRAIN/f'model_{n}.pt';assert p.is_file();protocol['权重'][str(n)]={'路径':str(p),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest()}
    path=OUT/'gpt-预定评估协议.json'
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(protocol))
    else:write(path,protocol)
    jobs=[]
    for n in ITERATIONS:
        for seed in SEEDS:
            for scene in SCENES:
                reused=False;run=OUT/f'{n}次'/f'种子{seed}'/'开启高程图'/scene
                if seed==46 and n in (2999,30000):
                    prefix='gpt-2999与30000权重对照-20261022/2999次' if n==2999 else 'gpt-关闭高程图对照-20261022'
                    run=ROOT/'gpt-复现记录'/prefix/'开启高程图'/scene;reused=True
                    assert (run/'gpt-评估完成.json').is_file()
                jobs.append({'迭代':n,'种子':seed,'场景':scene,'目录':str(run),'复用':reused})
    write(OUT/'gpt-运行清单.json',jobs)
    return jobs

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=8);parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--gpus',default='4,5');parser.add_argument('--min-iteration',type=int,default=0);parser.add_argument('--batch-name',default='批量');a=parser.parse_args()
    jobs=prepare()
    if a.prepare_only:return
    jobs=[j for j in jobs if j['迭代']>=a.min_iteration]
    gpus=[int(x) for x in a.gpus.split(',')]
    todo=queue.Queue();done=[];failed=[];lock=threading.Lock()
    for j in jobs:
        if (Path(j['目录'])/'gpt-评估完成.json').exists():done.append(j)
        else:todo.put(j)
    print(f'总计{len(jobs)}，已完成/复用{len(done)}，待运行{todo.qsize()}，并发{a.workers}',flush=True)
    def worker(index):
        gpu=gpus[index%len(gpus)]
        while True:
            try:j=todo.get_nowait()
            except queue.Empty:return
            label=f"{j['迭代']}次-种子{j['种子']}-{j['场景']}";run=Path(j['目录']);start=time.monotonic();ok=False;attempts=[]
            for attempt in range(1,3):
                log=OUT/f'gpt-{label}-运行{attempt}.log'
                print('开始',label,'设备',gpu,'尝试',attempt,flush=True)
                with log.open('w') as f:
                    try:
                        result=subprocess.run([sys.executable,str(ROOT/'gpt-关闭高程图评估.py'),'--condition','开启高程图','--scene',j['场景'],'--seed',str(j['种子']),'--terrain-seed','46','--gpu',str(gpu),'--checkpoint',str(j['迭代']),'--output-root',str(OUT/f"{j['迭代']}次"/f"种子{j['种子']}")],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=900)
                        code=result.returncode
                    except subprocess.TimeoutExpired:code='超时'
                ok=code==0 and (run/'gpt-评估完成.json').exists();attempts.append({'次数':attempt,'退出码':code,'日志':str(log)})
                if ok:break
            state={**j,'成功完成':ok,'设备':gpu,'耗时秒':time.monotonic()-start,'进程记录':attempts}
            write(OUT/f'gpt-{label}-进程状态.json',state)
            with lock:
                (done if ok else failed).append(j)
                progress={'计划':len(jobs),'完成':len(done),'运行失败':len(failed),'未结束':len(jobs)-len(done)-len(failed),'最近完成':state}
                write(OUT/f'gpt-{a.batch_name}进度.json',progress)
                print('进度',len(done),'/',len(jobs),'运行失败',len(failed),label,round(state['耗时秒'],1),'秒',flush=True)
            todo.task_done()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(worker,i) for i in range(a.workers)]
        for f in concurrent.futures.as_completed(futures):f.result()
    write(OUT/f'gpt-{a.batch_name}执行结果.json',{'完成':len(done),'总计':len(jobs),'运行失败':failed})
    if failed:raise RuntimeError('有仿真运行失败，不计作策略失败；可断点重跑。')
    print(f'{len(jobs)}组全部完成',flush=True)
if __name__=='__main__':main()
