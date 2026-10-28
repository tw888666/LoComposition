"""绘制同一模型的配对仿真轨迹，保留原始10秒窗口。"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

root = Path(__file__).resolve().parent
out = root / "gpt-复现记录"
r = json.loads((out / "gpt-姿态速度原因排查-20261019.json").read_text())
font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False})
fig, axes = plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
for name,color in [("默认评估","#2563eb"),("训练电机","#ea580c"),("范围内横向指令","#16a34a")]:
    folder=Path(r[name]['输入目录']);a=np.load(folder/'plots/sim_data.npz')
    meta=json.loads((folder/'metrics_summary.json').read_text())
    # 完整场景最后一步发生重置，图中不将重置状态当作真实运动轨迹。
    time=np.arange(499)*.02
    if name!='范围内横向指令':
        axes[0].plot(time,a['base_linear_velocity_body_array'][500:999,0],label=name,color=color,alpha=.85)
    stop=min([v for v in meta['automatic_reset_steps'] if 1000<=v<1500] or [1500])
    ori=a['base_orientation_array'][1000:stop]
    angle=np.arccos(np.clip(np.cos(ori[:,1])*np.cos(ori[:,2]),-1,1))
    axes[1].plot(np.arange(len(angle))*.02,angle,label=name,color=color,alpha=.8,linewidth=1.1)
    if stop<1499:
        axes[1].axvline((stop-1000)*.02,color=color,linestyle=':',label='默认评估提前重置')
axes[0].axhline(1.6,color='black',linestyle='--',label='指令 1.6 米/秒')
axes[1].axhline(.1,color='black',linestyle='--',label='倾角阈值 0.1 弧度')
axes[0].set(title='平地高速跟踪：仅改变电机配置',xlabel='场景时间（秒）',ylabel='实际前向速度（米/秒）')
axes[1].set(title='崎岖地形斜向行走：电机或横向指令对照',xlabel='场景时间（秒）',ylabel='机身总倾角（弧度）')
for ax in axes:
    ax.axvspan(0,1,color='grey',alpha=.1)
    ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('同一最终模型、种子46；阴影为首秒，提前重置后的轨迹不计入固定场景',fontsize=12)
fig.savefig(out/'gpt-姿态速度原因排查-20261019.png',dpi=170)
print(out/'gpt-姿态速度原因排查-20261019.png')
