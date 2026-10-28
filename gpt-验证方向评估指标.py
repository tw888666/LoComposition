"""核验方向投影、跨角度边界、折返、无净进展和重置截断等指标边界。"""
from pathlib import Path
import importlib.util
import numpy as np

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('gpt_summary_metrics',ROOT/'gpt-四组微调结果汇总.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sample(progress, side=None, heading=0., yaw=None, pitch=0.):
    progress=np.asarray(progress,dtype=float)
    side=np.zeros_like(progress) if side is None else np.asarray(side,dtype=float)
    n=len(progress)
    f=np.array([np.cos(heading),np.sin(heading)])
    l=np.array([-np.sin(heading),np.cos(heading)])
    origin=np.array([7.,-4.,.4])
    positions=np.column_stack([origin[:2]+progress[:,None]*f+side[:,None]*l,np.full(n,.4)])
    yaws=np.full(n,heading if yaw is None else yaw)
    data={'机身位置':positions,'机身姿态偏航俯仰横滚':np.column_stack([yaws,np.full(n,pitch),np.zeros(n)]),
          '机身线速度机体系':np.tile([1.,0.,0.],(n,1)), '机身角速度机体系':np.zeros((n,3)),
          '关节力矩':np.ones((n,12)), '关节角速度':np.ones((n,12)), '指令速度':np.tile([1.,0.,0.],(n,1))}
    cfg={'初始航向弧度':heading,'起点':origin.tolist(),'控制周期秒':1.,'指令速度米每秒':1.}
    return data,cfg


count=0
for heading in (0.,np.pi/2,np.pi,-np.pi/3):
    data,cfg=sample([1,2,3],heading=heading)
    r=module.metrics(data,3,cfg)
    np.testing.assert_allclose([r['沿初始方向有效前进距离米'],r['最大绝对横向偏移米'],
        r['航向偏差均方根度'],r['沿初始方向速度均方根误差米每秒'],r['每米有效前进机械功焦耳']], [3,0,0,0,12],atol=1e-12)
    count+=1
data,cfg=sample([1,2,3],[1,2,3],yaw=np.pi/4)
data['机身线速度机体系'][:,0]=np.sqrt(2)
r=module.metrics(data,3,cfg)
np.testing.assert_allclose([r['航向偏差均方根度'],r['终点绝对横向偏移米'],r['沿初始方向有效前进距离米'],
    r['每米有效前进机械功焦耳'],r['每米绝对机械功焦耳'],r['沿初始方向速度均方根误差米每秒']],
    [45,3,3,12,12/np.sqrt(2),0],atol=1e-12)
count+=1
data,cfg=sample([1,2,1]); r=module.metrics(data,3,cfg)
np.testing.assert_allclose([r['沿初始方向有效前进距离米'],r['水平路径长度米'],r['每米有效前进机械功焦耳']],[1,3,36])
count+=1
for progress in ([1,0],[-1,-2]):
    data,cfg=sample(progress); r=module.direction_metrics(data,len(progress),cfg)
    assert r['每米有效前进机械功焦耳'] is None and not r['有效前进能耗可定义']
    count+=1
data,cfg=sample([1,2,100]);r=module.metrics(data,2,cfg)
np.testing.assert_allclose([r['沿初始方向有效前进距离米'],r['每米有效前进机械功焦耳']],[2,12])
count+=1
data,cfg=sample([1,2],heading=np.deg2rad(179),yaw=np.deg2rad(-179));r=module.direction_metrics(data,2,cfg)
np.testing.assert_allclose(r['航向偏差均方根度'],2,atol=1e-12)
count+=1
data,cfg=sample([1,2],pitch=np.pi/6);r=module.direction_metrics(data,2,cfg)
np.testing.assert_allclose(r['沿初始方向速度均方根误差米每秒'],1-np.cos(np.pi/6),atol=1e-12)
count+=1
data,cfg=sample([1,2]);del cfg['初始航向弧度']
try:
    module.direction_metrics(data,2,cfg)
except ValueError:
    count+=1
else:
    raise AssertionError('未知协议不能静默假定初始朝向。')
print(f'方向指标边界核验通过：{count}项。')
