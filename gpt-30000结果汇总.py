"""沿用参考对话的分析函数，核验30000轮模型完整评估及阶梯通过率。"""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'gpt-复现记录/gpt-30000模型对照评估-20261021'
STAIRS = OUT / '阶梯评估'

def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT/file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def main():
    stats = load('stair_stats', 'gpt-30000阶梯统计.py')
    analysis = load('full_stats', 'gpt-作者21349结果分析.py')
    subprocess.run([sys.executable, str(ROOT/'gpt-30000阶梯统计.py')], check=True)
    d = json.loads((STAIRS/'gpt-通过率统计.json').read_text())
    assert all(s['完成次数']==10 for s in d['统计'])
    assert len(d['两高度初始动力学配对核验'])==10
    assert all(all(r[k] for k in ('各刚体质量千克','材料属性','局部质心')) for r in d['两高度初始动力学配对核验'])
    low = stats.analyze_run(STAIRS/'8.5厘米/种子40')
    assert low is not None
    comparison = []
    full_results = {}
    sources = {
        '自训30000：取消质心偏移': next((OUT/'完整评估').glob('eval_*/metrics_summary.json')).parent,
        '作者21349：取消质心偏移': next((ROOT/'gpt-复现记录/gpt-21349历史配置核查-20261019/仅取消质心偏移').glob('eval_*/metrics_summary.json')).parent,
        '自训30000：原默认评估': ROOT/'locomposition/logs/clean_rl/gpt-go2-default-seed46/2026-10-10-09-56-51/eval_checkpoint_30000_seed_46_action_delay_0',
    }
    for label, path in sources.items():
        result, data = analysis.analyze(path)
        full_results[label] = result
        comparison.append({'模型与条件':label, '平地运输能耗':result['主区间运输能耗'],
                           '作者口径速度均方根误差米每秒':result['全窗口']['作者两轴RMSE平均米每秒'],
                           '补算五类最大软约束违规率百分比':result['全窗口']['论文五类最大违规率百分比'],
                           '固定场景完成数':result['固定场景完成数']})
    selected = []
    geometry = []
    for h in (8.5,15,20):
        rows = [low] if h==8.5 else [r for r in d['逐次结果'] if r['台阶厘米']==h]
        selected.append(min(rows, key=lambda r:r['种子']))
        for success in (True,False):
            group=[r for r in rows if r['通过']==success]
            if group:
                r=min(group,key=lambda r:r['种子'])
                if r not in selected:selected.append(r)
        ref = None
        for r in rows:
            run=Path(r['目录']);mesh=np.load(run/'gpt-实际地形.npz')
            if ref is None:ref={k:mesh[k].copy() for k in ('vertices','faces')}
            assert np.array_equal(mesh['faces'],ref['faces'])
            assert np.max(np.abs(mesh['vertices']-ref['vertices']))<1e-12
            summary=json.loads(next(run.glob('eval_*/metrics_summary.json')).read_text())
            assert summary['total_sim_steps']==750 and summary['seed']==r['种子']
            assert Path(summary['used_checkpoint_path']).resolve()==Path(json.loads((OUT/'gpt-任务与模型来源.json').read_text())['模型'])
            with np.load(r['轨迹']) as data:
                for key in ('base_position_array','joint_positions_array','base_orientation_array','commanded_velocity_array'):
                    assert len(data[key])==750 and np.isfinite(data[key]).all()
                np.testing.assert_allclose(data['commanded_velocity_array'],np.tile([1,0,0],(750,1)),atol=1e-6)
        # 从原世界坐标的爬楼通道采样台阶表面，确认实际每级高度。
        sys.path.insert(0, str(ROOT/'.gpt-video-deps'))
        import trimesh
        mesh=trimesh.Trimesh(vertices=ref['vertices'],faces=ref['faces'],process=False)
        xyz=np.array([[-8,16,3]]+[[-9.4-.3*i,16,3] for i in range(7)])
        hit,ids,_=mesh.ray.intersects_location(xyz,np.tile([0,0,-1],(len(xyz),1)),multiple_hits=False)
        z=np.full(len(xyz),np.nan);z[ids]=hit[:,2]
        np.testing.assert_allclose(np.diff(z),h/100,atol=1e-6)
        geometry.append({'台阶厘米':h,'地形高度采样米':z.tolist(),'核验':'同高度各评估种子的地形一致；七级实际高度与目标一致'})
    (OUT/'gpt-完整评估对照.json').write_text(json.dumps(full_results,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'gpt-模型指标对照.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=comparison[0]);w.writeheader();w.writerows(comparison)
    (OUT/'gpt-录像选例.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2)+'\n')
    (OUT/'gpt-最终核验.json').write_text(json.dumps({'阶梯统计':d['统计'],'低台阶对照':low,'地形核验':geometry,
       '两高度动力学配对':d['两高度初始动力学配对核验'],'模型':json.loads((OUT/'gpt-任务与模型来源.json').read_text()),
       '脚本SHA256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('gpt-30000*.py')}},ensure_ascii=False,indent=2)+'\n')
    ref=json.loads((ROOT/'gpt-复现记录/gpt-21349高台阶多种子-20261019/gpt-通过率统计.json').read_text())
    lines=['自训30000轮模型：参考21349对话的完整评估、阶梯通过率与录像','日期：2026-10-21','',
        '完整评估：取消质心随机偏移；种子46；4000随机步+21个场景各500步，合计14500控制步。',
        '速度口径：(前向均方根误差+横向均方根误差)/2；能耗：0.6–1.6米/秒六点等权平均。',
        '五类软约束为补算指标，姿态取总倾角0.1弧度；不把作者脚本的全部约束最大频率混作此指标。','']
    lines += [json.dumps(row,ensure_ascii=False) for row in comparison]
    lines += ['', '阶梯协议：七级、踏面30厘米、地形种子46，前向指令1米/秒，最长15秒，评估种子40–49。',
        '成功：首次提前重置前，四足X≤-11.05、Z≥-0.03、Y在13.5–18.5，机身Z≥0.2且倾角≤30度，连续10步。',
        '失败后随机续跑不计入通过判定；仍站立但未爬出亦为失败。低台阶只作额外对照。','']
    for s in d['统计']:
        old=next(t for t in ref['统计'] if t['台阶厘米']==s['台阶厘米'])
        rows=[r for r in d['逐次结果'] if r['台阶厘米']==s['台阶厘米']]
        reasons={key:sum(r['结果']==key for r in rows) for key in sorted({r['结果'] for r in rows})}
        lo,hi=s['威尔逊95%置信区间']
        lines.append(f"{s['台阶厘米']}厘米：自训30000为{s['通过次数']}/10，作者21349为{old['通过次数']}/10；当前95%威尔逊区间{lo:.1%}–{hi:.1%}。")
        lines.append('结果分类：'+json.dumps(reasons,ensure_ascii=False))
    lines += [f"8.5厘米种子40：{low['结果']}；通过时间：{low['通过时间秒']}秒。",'',
        '录像按每高度最低种子及最低成功/失败种子选取；没有成功时如实标注，不挪用其他模型成功片段。',
        '录像为真实仿真轨迹和直接导出地形的运动学回放，MuJoCo（仿真与渲染工具）仅渲染，不重新推理或积分。',
        '局限：一个训练种子，多个评估种子，不等于多个独立训练结果。对照两模型加载路径不同，未保证逐步噪声完全配对。',
        '当前是统一测试条件的模型比较，不是论文主表统计复现。完整场景跑满不等于所有速度/姿态达标。',
        '下一步：在固定电机配置下做较低速度的阶梯对照，或对齐训练范围后进行配对训练。']
    (OUT/'gpt-评估结果说明.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'完整评估':comparison,'阶梯':d['统计'],'低台阶':low,'录像目录':[r['目录'] for r in selected]},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
