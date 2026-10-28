"""复算固定楼梯轨迹的航向与侧偏，保留原正式评估结果不变。"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'评估/gpt-逐次评估指标.json').read_text())
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family']=font.get_name()
plt.rcParams['axes.unicode_minus']=False
colors={'A0':'#687786','A':'#3973ac','B':'#dc9136','C':'#3f8d6d','D':'#966fa8'}
fig,axes=plt.subplots(1,3,figsize=(15,4.4))
records=[]
for label in colors:
 for seed in (46,47,48):
  r=next(x for x in rows if x['组别']==label and x['种子']==seed and x['场景']=='楼梯8.5厘米')
  run=Path(r['目录']);d=np.load(run/'gpt-控制步真实数据.npz')
  cfg=json.loads((run/'gpt-评估条件.json').read_text())
  dt=cfg['控制周期秒']; t=np.arange(1,len(d['机身位置'])+1)*dt
  yaw=d['机身姿态偏航俯仰横滚'][:,0].astype(float)
  delta=np.rad2deg(np.unwrap(np.arctan2(np.sin(yaw-np.pi),np.cos(yaw-np.pi))))
  gate=round(r['越过平台秒']/dt)-1
  pos=d['机身位置']; side=pos[:,1]-cfg['起点'][1]; progress=cfg['起点'][0]-pos[:,0]
  cmd=np.unique(d['指令速度'],axis=0)
  assert cmd.shape==(1,3) and np.array_equal(cmd[0],[1,0,0])
  records.append({'组别':label,'种子':seed,'通过时刻秒':r['越过平台秒'],
     '通过时航向偏差度':float(delta[gate]),'终点航向偏差度':float(delta[-1]),
     '最大绝对航向偏差度':float(abs(delta).max()),'终点绝对侧偏米':float(abs(side[-1])),
     '前向速度均方根误差米每秒':r['自身窗口指标']['前向速度均方根误差米每秒'],
     '偏航角速度均方根误差弧度每秒':r['自身窗口指标']['偏航角速度均方根误差弧度每秒'],
     '全程唯一速度指令':cmd[0].tolist(),'源目录':str(run)})
  if seed!=46:continue
  axes[0].plot(progress,side,label=label,color=colors[label],lw=2)
  axes[0].scatter(progress[gate],side[gate],color=colors[label],s=30)
  axes[1].plot(t,delta,label=label,color=colors[label],lw=2)
  axes[1].scatter(t[gate],delta[gate],color=colors[label],s=30)
  if label=='B':
   axes[2].plot(t,d['机身线速度机体系'][:,0],color=colors[label],lw=1.5,label='B：机身前向实际速度')
   axes[2].axhline(1,color='#333333',ls='--',label='固定指令 1 米/秒')
   axes[2].axvline(t[gate],color='#777777',ls=':',label=f'越阶判定 {t[gate]:.2f} 秒')
axes[0].axhline(0,color='#777777',ls='--',lw=1)
axes[0].set(xlabel='沿初始朝向的前进距离（米）',ylabel='世界Y方向侧偏（米）',title='真实俯视轨迹：B逐渐偏离直线')
axes[0].set_aspect('equal',adjustable='datalim')
axes[1].axhline(0,color='#777777',ls='--',lw=1)
axes[1].set(xlabel='时间（秒）',ylabel='相对初始目标朝向的偏差（度）',title='B在通过台阶前已明显偏转')
axes[2].set(xlabel='时间（秒）',ylabel='机身前向速度（米/秒）',title='前向速度误差不直接衡量朝向偏离')
for ax in axes:
 ax.grid(alpha=.2);ax.legend(fontsize=8,loc='best')
fig.suptitle('8.5厘米楼梯 · 视频对应种子46 · 圆点表示首次满足越阶判据',fontsize=14)
fig.tight_layout()
fig.savefig(OUT/'gpt-楼梯轨迹与航向核查.png',dpi=170)
plt.close(fig)
(OUT/'gpt-楼梯航向补充指标.json').write_text(json.dumps({'定义':'航向偏差相对固定初始目标180度；侧偏相对起始世界Y=16米。','判据限制':'原楼梯成功仅表示越阶并连续直立，不要求通过后保持初始航向或全程侧偏不超过1米。','指标':records},ensure_ascii=False,indent=2))
print(OUT/'gpt-楼梯轨迹与航向核查.png')
