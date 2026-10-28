"""分析原评估及三组配对诊断，分别报告完整10秒和1至10秒诊断窗口。"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "gpt-复现记录"
BASE = ROOT / "locomposition/logs/clean_rl/gpt-go2-default-seed46/2026-10-10-09-56-51"
cases = {"原始完整评估": BASE / "eval_checkpoint_30000_seed_46_action_delay_0"}
for condition in ["默认评估", "训练电机", "范围内横向指令", "复测-默认评估", "复测-训练电机"]:
    paths = list((OUT / "gpt-姿态速度诊断-20261019" / condition).glob("eval_*/metrics_summary.json"))
    if paths: cases[condition] = paths[0].parent

shared_stop = 1499
if '默认评估' in cases:
    baseline_meta=json.loads((cases['默认评估']/'metrics_summary.json').read_text())
    shared_stop=min([v for v in baseline_meta['automatic_reset_steps'] if 1000<=v<1500] or [1500])
result = {}
for condition, folder in cases.items():
    meta = json.loads((folder / "metrics_summary.json").read_text())
    a = np.load(folder / "plots/sim_data.npz", allow_pickle=False)
    n = meta["total_sim_steps"]
    valid = np.ones(n, bool)
    valid[meta["automatic_reset_steps"]] = False
    ori = a["base_orientation_array"]
    tilt = np.arccos(np.clip(np.cos(ori[:,1])*np.cos(ori[:,2]), -1, 1))
    cmd = a["commanded_velocity_array"]
    vel = a["base_linear_velocity_body_array"]
    diagnostic_path = folder.parent / 'gpt-电机与姿态诊断.npz'
    diagnostic = np.load(diagnostic_path) if diagnostic_path.exists() else None
    def stats(mask):
        assert mask.any()
        v=vel[mask,:2]; c=cmd[mask,:2]; angle=tilt[mask]
        row = {
            "样本数": int(mask.sum()), "平均指令xy": c.mean(axis=0).tolist(),
            "实际平均速度xy": v.mean(axis=0).tolist(),
            "平均速度偏差xy": (v-c).mean(axis=0).tolist(),
            "速度RMSE分量xy": np.sqrt(np.mean((v-c)**2,axis=0)).tolist(),
            "总倾角违规率百分比": float((angle>.1).mean()*100),
            "重力投影违规率百分比": float((np.sin(angle)>.1).mean()*100),
            "最大单轴违规率百分比": float((np.abs(np.arctan2(np.sin(ori[mask,1:]),np.cos(ori[mask,1:])))>.1).mean(axis=0).max()*100),
            "平均总倾角弧度": float(angle.mean()), "倾角95分位弧度": float(np.quantile(angle,.95)),
            "最大绝对关节力矩牛米": float(np.abs(a['joint_torques_array'][mask]).max()),
            "最大绝对关节速度弧度每秒": float(np.abs(a['joint_velocities_array'][mask]).max()),
            "前向指令超训练范围百分比": float(((c[:,0]>1+1e-6)|(c[:,0]<-.3-1e-6)).mean()*100),
            "横向指令超训练范围百分比": float((np.abs(c[:,1])>.7+1e-6).mean()*100),
        }
        if diagnostic is not None:
            delta = np.abs(diagnostic['computed_effort'][mask] - diagnostic['applied_effort'][mask])
            row['控制步末任一电机限幅时间百分比'] = float((delta>1e-3).any(axis=1).mean()*100)
            row['控制步末电机限幅关节样本百分比'] = float((delta>1e-3).mean()*100)
            row['控制步末限幅平均绝对力矩差牛米'] = float(delta.mean())
        return row
    rows=[]
    for i,(name,*_) in enumerate(meta['fixed_command_scenarios']):
        start=meta['random_sim_steps']+i*meta['fixed_command_sim_steps'];end=start+meta['fixed_command_sim_steps']
        whole=(np.arange(n)>=start)&(np.arange(n)<end)
        steady=whole&(np.arange(n)>=start+50)&valid
        resets=[v for v in meta['automatic_reset_steps'] if start<=v<end]
        first_reset=min(resets or [end])
        row={"场景":name,"完整10秒":stats(whole),"去首秒及自动重置步":stats(steady),
             "前1秒":stats(whole&(np.arange(n)<start+50)),
             "自动重置步骤":resets,"发生提前重置":first_reset<end-1,
             "首次重置相对时间秒":(first_reset-start)*.02,
             "去首秒且仅首次重置前":stats(steady&(np.arange(n)<first_reset)),
             "逐秒总倾角超限百分比":[float((tilt[start+j*50:start+(j+1)*50]>.1).mean()*100) for j in range(10)]}
        if condition!='原始完整评估' and i==2:
            row['配对共同窗口1至最早重置']=stats(steady&(np.arange(n)<shared_stop))
        rows.append(row)
    if meta['random_sim_steps']:
        mask=(np.arange(n)<meta['random_sim_steps'])&valid
        random=stats(mask)
    else:random=None
    result[condition]={"场景":rows,"随机指令段去自动重置步":random,"输入目录":str(folder)}
    if diagnostic is not None:
        assert len(diagnostic['gravity']) == n
        direct = np.linalg.norm(diagnostic['gravity'][:,:2],axis=1)
        result[condition]['重力投影与欧拉角重建最大绝对差'] = float(np.max(np.abs(direct-np.sin(tilt))))
        result[condition]['限幅记录频率说明'] = '每个50赫兹控制步的末物理子步；不等同于全部200赫兹物理步限幅率。'

if '默认评估' in result and '范围内横向指令' in result:
    # 命令干预只在第三个场景发生；前两个场景应作为数值重复性对照。
    b=list((OUT/'gpt-姿态速度诊断-20261019/默认评估').glob('eval_*/plots/sim_data.npz'))[0]
    c=list((OUT/'gpt-姿态速度诊断-20261019/范围内横向指令').glob('eval_*/plots/sim_data.npz'))[0]
    ab=np.load(b);ac=np.load(c)
    result['配对重复性检查']={k:float(np.max(np.abs(ab[k][:1000]-ac[k][:1000]))) for k in ['commanded_velocity_array','base_linear_velocity_body_array','base_orientation_array']}

p=OUT/'gpt-姿态速度原因排查-20261019.json';p.write_text(json.dumps(result,ensure_ascii=False,indent=2))
for name,case in result.items():
    if name=='配对重复性检查':print(name,case);continue
    print(name)
    for row in case['场景']:
        if row['场景'] in ['walk_x_flat_terrain_1.0mps','walk_x_flat_terrain_1.6mps','fast_walk_diagonal_uneven_terrain','medium_walk_diagonal_turning_uneven_terrain','medium_walk_diagonal_random_steps']:
            view=row.get('配对共同窗口1至最早重置',row['去首秒且仅首次重置前'])
            print(row['场景'],json.dumps({'提前重置':row['发生提前重置'],'时间':row['首次重置相对时间秒'],'实际速度':view['实际平均速度xy'],'姿态违规率':view['总倾角违规率百分比'],'限幅率':view.get('控制步末任一电机限幅时间百分比')},ensure_ascii=False))
