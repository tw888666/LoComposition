"""按预先定义的越阶规则统计成功率，排除自动重置后的续跑。"""
from pathlib import Path
import argparse,csv,hashlib,json,math
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'gpt-复现记录/gpt-21349高台阶多种子-20261019'
PROTOCOL={'种子清单':list(range(40,50)),'台阶高度厘米':[15,20],'级数':7,'踏面深度米':.3,'地形种子':46,'时间上限秒':15,'速度指令米每秒':1.,'成功定义':{'四足世界X均不大于':-11.05,'四足世界Z均不低于':-.03,'四足世界Y范围':[13.5,18.5],'机身世界Z不低于':.2,'机身倾斜角上限度':30,'连续控制步数':10,'控制周期秒':.02,'说明':'以上条件在首次提前重置前连续满足10步。到达平台后在相邻地形跌倒不撤销本次爬楼成功；另行记录后续重置。'},'失败定义':'到15秒未通过，或通过前发生环境提前终止。不能把存活到超时当作通过。','录像选例规则':'每种高度优先选择最低种子的成功和失败案例，不根据动作美观挑选；没有某类结果时如实说明。','模型':'2025-06-28-17-13-04_21349_traced_deterministic.pt','配置':'仅取消质心随机偏移；固定地形布局，只调整倒金字塔楼梯高度；重置阶段直接定位起点并设置固定速度，避免首帧观测位置不一致；地形构造后重设评估种子以配对动力学随机化。'}

def classify(data,summary):
    n=len(data['base_position_array']);first_reset=min(summary['automatic_reset_steps'],default=n)
    end=min(n,first_reset) # 重置步的post-step数据来自下一回合，永不参与通过判定。
    pos=data['base_position_array'][:end];feet=data['foot_positions_world_frame_array'][:end];orientation=data['base_orientation_array'][:end]
    upright=np.cos(orientation[:,1])*np.cos(orientation[:,2])
    gate=(np.all(feet[:,:,0]<=-11.05,axis=1)&np.all(feet[:,:,2]>=-.03,axis=1)&np.all((feet[:,:,1]>=13.5)&(feet[:,:,1]<=18.5),axis=1)&(pos[:,2]>=.2)&(upright>=np.cos(np.pi/6)))
    windows=np.convolve(gate.astype(int),np.ones(10,dtype=int),'valid') if len(gate)>=10 else np.array([])
    done=np.flatnonzero(windows==10);step=int(done[0]+9) if len(done) else None
    return step,end,gate

def wilson(k,n):
    if n==0:return [None,None]
    z=1.959963984540054;den=1+z*z/n;c=(k/n+z*z/(2*n))/den;r=z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/den
    return [0. if k==0 else max(0,c-r),1. if k==n else min(1,c+r)]

def analyze_run(run):
    files=list(run.glob('eval_*/metrics_summary.json'))
    if not files:return None
    summary=json.loads(files[0].read_text());source=files[0].parent/'plots/sim_data.npz';data=np.load(source)
    cond=json.loads((run/'gpt-评估条件.json').read_text());step,end,gate=classify(data,summary)
    resets=json.loads((run/'gpt-重置前真实状态.json').read_text()) if (run/'gpt-重置前真实状态.json').exists() else []
    first=resets[0] if resets else None
    reasons=[k for k,v in first['终止项'].items() if v and k!='time_out'] if first else []
    success=step is not None
    reason='通过' if success else ('提前终止：'+','.join(reasons) if reasons else '15秒内未满足越阶条件')
    positions=data['base_position_array'][:end];q=positions[-1] if len(positions) else np.full(3,np.nan)
    rng=json.loads((run/'gpt-运行时随机化.json').read_text())
    record={'台阶厘米':cond['台阶高度米']*100,'种子':cond['评估种子'],'通过':success,'结果':reason,'通过时间秒':(step+1)*.02 if success else None,'有效记录时长秒':end*.02,'首次重置时刻秒':first['全局步']*.02 if first else None,'首次重置原因':reasons or ['time_out'],'最大前进米':float(-8-np.min(positions[:,0])) if len(positions) else 0,'有效末帧机身位置':q.tolist(),'总质量千克':sum(rng['各刚体质量千克']),'首个通过步':step,'首次重置前记录步数':end,'轨迹':str(source),'轨迹SHA256':hashlib.sha256(source.read_bytes()).hexdigest(),'目录':str(run)}
    (run/'gpt-通过判定.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    return record

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--init',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=OUT/'gpt-预定评估协议.json'
    if not protocol.exists():protocol.write_text(json.dumps(PROTOCOL,ensure_ascii=False,indent=2)+'\n')
    if args.init:return
    rows=[]
    for h in (15,20):
        for seed in range(40,50):
            r=analyze_run(OUT/f'{h}厘米'/f'种子{seed}')
            if r:rows.append(r)
    stats=[]
    for h in (15,20):
        group=[r for r in rows if r['台阶厘米']==h];k=sum(r['通过'] for r in group);n=len(group)
        stats.append({'台阶厘米':h,'完成次数':n,'计划次数':10,'通过次数':k,'通过率':k/n if n else None,'威尔逊95%置信区间':wilson(k,n),'成功种子':[r['种子'] for r in group if r['通过']],'失败种子':[r['种子'] for r in group if not r['通过']]})
    paired=[]
    for seed in range(40,50):
        a=OUT/'15厘米'/f'种子{seed}'/'gpt-运行时随机化.json';b=OUT/'20厘米'/f'种子{seed}'/'gpt-运行时随机化.json'
        if a.exists() and b.exists():
            av=json.loads(a.read_text());bv=json.loads(b.read_text());paired.append({'种子':seed,**{k:av[k]==bv[k] for k in ('各刚体质量千克','材料属性','局部质心')}})
    result={'统计':stats,'逐次结果':rows,'两高度初始动力学配对核验':paired}
    (OUT/'gpt-通过率统计.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    columns=['台阶厘米','种子','通过','结果','通过时间秒','有效记录时长秒','最大前进米','总质量千克']
    with (OUT/'gpt-逐种子结果.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    print(json.dumps({'统计':stats,'配对核验':paired},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
