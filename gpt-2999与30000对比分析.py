"""比较2999与30000次权重，核验两组环境一致并统计首次重置前表现。"""
from pathlib import Path
import csv,hashlib,importlib.util,json
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'gpt-复现记录/gpt-2999与30000权重对照-20261022'
BASELINE=ROOT/'gpt-复现记录/gpt-关闭高程图对照-20261022/开启高程图'
SCENES=('平地1.0','平地1.6','崎岖地形','8.5厘米楼梯')
spec=importlib.util.spec_from_file_location('observation_analysis',ROOT/'gpt-关闭高程图分析.py')
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def metrics(data,length):
    err=data['base_linear_velocity_body_array'][:length,:2]-data['commanded_velocity_array'][:length,:2]
    ang=data['base_orientation_array'][:length]
    tilt=np.arccos(np.clip(np.cos(ang[:,1])*np.cos(ang[:,2]),-1,1))
    power=np.abs(data['joint_torques_array'][:length]*data['joint_velocities_array'][:length]).sum(axis=1).astype(float)
    dist=float(data['distance_increment_array'][:length].sum())
    return {'平均前向速度米每秒':float(data['base_linear_velocity_body_array'][:length,0].mean()),
        '速度均方根误差米每秒':float(np.sqrt(np.mean(err.astype(float)**2,axis=0)).mean()),
        '前向速度均方根误差米每秒':float(np.sqrt(np.mean(err[:,0].astype(float)**2))),
        '总倾角超0.1弧度百分比':float((tilt>.1).mean()*100),
        '总倾角均方根度':float(np.rad2deg(np.sqrt(np.mean(tilt**2)))),
        '关节绝对机械功率均值瓦':float(power.mean()),'关节绝对机械功焦耳':float(power.sum()*.02),
        '水平路径长度米':dist,'每米关节绝对机械功焦耳':float(power.sum()*.02/dist) if dist>1e-9 else None}
def get_record(run,iteration):
    if iteration==2999:r=analysis.analyze(run)
    else:r=json.loads((run/'gpt-场景结果.json').read_text())
    if r is None:return None
    r=dict(r);r['迭代']=iteration
    cfg=json.loads((run/'gpt-评估条件.json').read_text())
    assert cfg['条件']=='开启高程图' and Path(cfg['权重']).name==f'model_{iteration}.pt'
    summary=json.loads(next(run.glob('eval_*/metrics_summary.json')).read_text())
    assert Path(summary['used_checkpoint_path']).resolve()==Path(cfg['权重']).resolve()
    assert hashlib.sha256(Path(cfg['权重']).read_bytes()).hexdigest()==cfg['权重SHA256']
    assert hashlib.sha256(Path(r['轨迹']).read_bytes()).hexdigest()==r['轨迹SHA256']
    with np.load(run/'gpt-实际策略输入.npz') as audit:
        np.testing.assert_array_equal(audit['原始观测'],audit['实际输入'])
        assert audit['实际输入'].shape==(500,236) and np.isfinite(audit['实际输入']).all()
    d=np.load(r['轨迹']);r.update(metrics(d,r['首次重置前记录步数']))
    r['权重SHA256']=cfg['权重SHA256'];r['复用已完成的30000次评估']=iteration==30000
    if iteration==2999:save(run/'gpt-场景结果.json',r)
    return r

def main():
    rows=[];checks=[]
    for scene in SCENES:
        early=OUT/'2999次/开启高程图'/scene;late=BASELINE/scene
        if not (early/'gpt-评估完成.json').exists():continue
        pair=[get_record(early,2999),get_record(late,30000)];rows.extend(pair)
        assert (early/'gpt-构造环境配置.yaml').read_bytes()==(late/'gpt-构造环境配置.yaml').read_bytes(),scene+'环境配置不一致'
        assert json.loads((early/'gpt-运行时随机化.json').read_text())==json.loads((late/'gpt-运行时随机化.json').read_text()),scene+'初始动力学不一致'
        with np.load(early/'gpt-实际地形.npz') as a,np.load(late/'gpt-实际地形.npz') as b:
            np.testing.assert_array_equal(a['faces'],b['faces']);np.testing.assert_allclose(a['vertices'],b['vertices'],atol=1e-12,rtol=0)
        with np.load(early/'gpt-实际策略输入.npz') as a,np.load(late/'gpt-实际策略输入.npz') as b:
            np.testing.assert_array_equal(a['原始观测'][0],b['原始观测'][0])
        n=min(r['首次重置前记录步数'] for r in pair)
        common={str(r['迭代']):metrics(np.load(r['轨迹']),n) for r in pair}
        checks.append({'场景':scene,'环境配置一致':True,'初始动力学一致':True,'地形一致':True,'首步原始观测一致':True,'两组真实高程输入正常':True,'实际加载权重核验':True,'共同有效窗口秒':n*.02,'共同窗口指标':common})
    result={'完整':len(rows)==8,'场景结果':rows,'配对核验':checks,'比较说明':'相同种子46、每场景10秒，两组均开启真实高程图，使用各自权重保存的观测归一化参数。30000次复用同日已核验的四场景结果。'}
    save(OUT/'gpt-权重对比汇总.json',result)
    columns=['场景','迭代','结果','平均前向速度米每秒','前向速度均方根误差米每秒','速度均方根误差米每秒','前向位移米','有效时长秒','楼梯通过时间秒','总倾角均方根度','总倾角超0.1弧度百分比','关节绝对机械功率均值瓦','每米关节绝对机械功焦耳']
    with (OUT/'gpt-逐场景权重对比.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    lines=['2999与30000次训练权重对照：两组均开启高程图','',result['比较说明'],
        '四场景：平地指令1.0米/秒、平地指令1.6米/秒、崎岖地形指令1.0米/秒、8.5厘米七级楼梯指令1.0米/秒。',
        '评估取消质心随机偏移；其余随机化沿用上一轮对照。实际地形、初始动力学和首步原始观测逐一配对核验。',
        '只统计首次重置前轨迹。若提前终止，录像显示实际终止状态后定格，不拼接重置后的续跑。',
        '楼梯通过采用四足越过平台边界且保持直立连续10步的既定判据；10秒未通过不能解释为无限时间下无法通过。',
        '速度误差：同时报告前向单轴均方根误差，以及前向/侧向两轴均方根误差的平均。',
        '能量指标为各关节|力矩×角速度|之和积分，不包含电机电损耗，不能当作电池耗电量。',
        '倾角包含正常爬楼所需的身体倾斜，倾角较大不等于跌倒。速度不同也会影响姿态与功率。',
        '这是单一评估种子的四个有限场景对照，不是整个训练过程的性能曲线或多种子通过率。','']
    for r in rows:
        lines.append(f"{r['场景']} | {r['迭代']}次：{r['结果']}，平均前向速度{r['平均前向速度米每秒']:.3f}米/秒，前向速度误差{r['前向速度均方根误差米每秒']:.3f}米/秒，平均绝对机械功率{r['关节绝对机械功率均值瓦']:.2f}瓦，楼梯通过时间{r['楼梯通过时间秒']}秒。")
    lines+=['','进度：'+('四个场景配对核验全部通过。' if result['完整'] else '部分场景尚未完成。')]
    (OUT/'gpt-评估说明.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps([{k:r[k] for k in columns} for r in rows],ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
