"""核验实际送入策略的高程遮蔽及配对配置，统计首次重置前的表现。"""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'gpt-复现记录/gpt-关闭高程图对照-20261022'
SCENES=('平地1.0','平地1.6','崎岖地形','8.5厘米楼梯')
spec=importlib.util.spec_from_file_location('stair_stats',ROOT/'gpt-30000阶梯统计.py')
stairs=importlib.util.module_from_spec(spec);spec.loader.exec_module(stairs)

def analyze(run):
    if not (run/'gpt-评估完成.json').exists():return None
    cfg=json.loads((run/'gpt-评估条件.json').read_text())
    summarypath=next(run.glob('eval_*/metrics_summary.json'))
    s=json.loads(summarypath.read_text());source=summarypath.parent/'plots/sim_data.npz'
    d=np.load(source);audit=np.load(run/'gpt-实际策略输入.npz')
    raw=audit['原始观测'];actual=audit['实际输入']
    assert raw.shape==actual.shape==(500,236)
    assert np.isfinite(raw).all() and np.isfinite(actual).all()
    np.testing.assert_array_equal(raw[:,:93],actual[:,:93])
    if cfg['条件']=='关闭高程图':np.testing.assert_array_equal(actual[:,93:],np.full((500,143),-.30,dtype=actual.dtype))
    else:np.testing.assert_array_equal(actual,raw)
    for k in ('base_position_array','base_orientation_array','joint_positions_array','commanded_velocity_array'):
        assert len(d[k])==500 and np.isfinite(d[k]).all()
    np.testing.assert_allclose(d['commanded_velocity_array'],np.tile([cfg['指令速度米每秒'],0,0],(500,1)),atol=1e-6)
    resets=json.loads((run/'gpt-重置前真实状态.json').read_text()) if (run/'gpt-重置前真实状态.json').exists() else []
    end=min(s['automatic_reset_steps'],default=500)
    assert end>0
    first=resets[0] if resets else None
    reasons=[n for n,v in first['终止项'].items() if v and n!='time_out'] if first else []
    pos=d['base_position_array'][:end];ang=d['base_orientation_array'][:end]
    tilt=np.arccos(np.clip(np.cos(ang[:,1])*np.cos(ang[:,2]),-1,1))
    error=d['base_linear_velocity_body_array'][:end,:2]-d['commanded_velocity_array'][:end,:2]
    passstep=stairs.classify(d,s)[0] if cfg['场景']=='8.5厘米楼梯' else None
    result='提前终止：'+','.join(reasons) if reasons else '完成10秒，无提前终止'
    if cfg['场景']=='8.5厘米楼梯':result=('通过楼梯' if passstep is not None else '未通过楼梯')+'；'+result
    rec={'条件':cfg['条件'],'场景':cfg['场景'],'种子':cfg['种子'],'指令速度米每秒':cfg['指令速度米每秒'],
        '结果':result,'提前终止':bool(reasons),'首次重置前记录步数':end,'有效时长秒':end*.02,
        '首次终止时间秒':first['全局步']*.02 if first and reasons else None,'首次终止原因':reasons,
        '平均前向速度米每秒':float(d['base_linear_velocity_body_array'][:end,0].mean()),
        '前向位移米':float(-(pos[-1,0]-pos[0,0])),
        '速度均方根误差米每秒':float(np.sqrt((error.astype(float)**2).mean(axis=0)).mean()),
        '总倾角超0.1弧度百分比':float((tilt>.1).mean()*100),
        '楼梯通过时间秒':(passstep+1)*.02 if passstep is not None else None,
        '轨迹':str(source),'轨迹SHA256':hashlib.sha256(source.read_bytes()).hexdigest(),
        '地形SHA256':hashlib.sha256((run/'gpt-实际地形.npz').read_bytes()).hexdigest(),
        '实际策略输入核验':'通过','高程图实际均值米':float(actual[:,93:].mean()),
        '高程图逐帧空间标准差均值米':float(actual[:,93:].std(axis=1).mean()),'目录':str(run)}
    (run/'gpt-场景结果.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2)+'\n')
    return rec

def main():
    rows=[];checks=[]
    for scene in SCENES:
        for condition in ('开启高程图','关闭高程图'):
            r=analyze(OUT/condition/scene)
            if r:rows.append(r)
        a=OUT/'开启高程图'/scene;b=OUT/'关闭高程图'/scene
        if not all((run/'gpt-评估完成.json').exists() for run in (a,b)):continue
        assert (a/'gpt-构造环境配置.yaml').read_bytes()==(b/'gpt-构造环境配置.yaml').read_bytes()
        assert json.loads((a/'gpt-运行时随机化.json').read_text())==json.loads((b/'gpt-运行时随机化.json').read_text())
        with np.load(a/'gpt-实际地形.npz') as x,np.load(b/'gpt-实际地形.npz') as y:
            np.testing.assert_array_equal(x['faces'],y['faces']);np.testing.assert_allclose(x['vertices'],y['vertices'],atol=1e-12,rtol=0)
        x=np.load(a/'gpt-实际策略输入.npz');y=np.load(b/'gpt-实际策略输入.npz')
        np.testing.assert_array_equal(x['原始观测'][0],y['原始观测'][0])
        pair=[r for r in rows if r['场景']==scene];length=min(r['首次重置前记录步数'] for r in pair)
        common={}
        for r in pair:
            d=np.load(r['轨迹']);err=d['base_linear_velocity_body_array'][:length,:2]-d['commanded_velocity_array'][:length,:2]
            common[r['条件']]={'平均前向速度米每秒':float(d['base_linear_velocity_body_array'][:length,0].mean()),'速度均方根误差米每秒':float(np.sqrt((err.astype(float)**2).mean(axis=0)).mean())}
        checks.append({'场景':scene,'配置与初始动力学一致':True,'真实地形一致':True,'首步完整原始观测一致':True,'共同有效窗口秒':length*.02,'共同窗口指标':common})
    result={'场景结果':rows,'配对核验':checks,'完整':len(rows)==8}
    (OUT/'gpt-评估汇总.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    columns=['场景','条件','结果','平均前向速度米每秒','前向位移米','有效时长秒','楼梯通过时间秒','速度均方根误差米每秒','总倾角超0.1弧度百分比']
    with (OUT/'gpt-逐场景对比.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)
    lines=['第30000轮权重：关闭真实高程图后的行走对照','',
        '关闭定义：归一化前的143维高程输入全部替换为固定-0.30米，不根据真实地形或当前机身位置更新；其余93维保持不变。',
        '固定值沿用仓库部署配置hardcoded_elevation=-0.3。此实验是已训练感知策略的输入扰动，不是从头训练的盲行策略。',
        '四场景均为种子46、每场景10秒；相同权重、初始状态、真实地形、动力学参数和观测噪声设置。',
        '为核对关闭前后差异，本次两组重新配对评估；重置时设置起点和固定指令。楼梯为8.5厘米×7级。不能与前次不同初始化的轨迹逐帧等同。',
        '保留观测计算用于审计及维持随机数消耗，但真实高度不会传入关闭组策略。遮蔽发生于原观测噪声之后、策略归一化之前。',
        '只统计首次重置前状态；录像提前失败后展示终止前真实状态并定格，绝不使用重置后的续跑冒充原场景。',
        '本次为单模型、单评估种子的有限场景观察；结果不能推广为所有地形，也不能证明实机表现。','']
    for r in rows:lines.append(f"{r['场景']} | {r['条件']}：{r['结果']}，平均前向速度{r['平均前向速度米每秒']:.3f}米/秒，前进{r['前向位移米']:.2f}米，楼梯通过时间{r['楼梯通过时间秒']}秒。")
    lines+=['','进度：'+('全部八组完成，实际输入与配对核验通过。' if result['完整'] else '尚有场景未完成。'),
            '下一步：若需要验证感知依赖的稳定性，可扩大随机种子，或改变固定高度值；当前未执行该扩展。']
    (OUT/'gpt-评估说明.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps([{k:r[k] for k in columns} for r in rows],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
