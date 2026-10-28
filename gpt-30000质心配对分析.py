"""核验质心配对条件，按种子配对统计速度、姿态和任务完成情况。"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'gpt-复现记录/gpt-30000质心多种子配对评估-20261026'
SCENES = ('平地1.0', '平地1.6', '崎岖地形', '8.5厘米楼梯')
CONDITIONS = ('取消质心偏移', '保留质心偏移')
spec = importlib.util.spec_from_file_location('stairs', ROOT / 'gpt-30000阶梯统计.py')
stairs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stairs)

def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def cfg(path):
    return yaml.safe_load(re.sub(r'!!python\S*', '', Path(path).read_text()))

def difference(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(set(a) | set(b)):
            result += difference(a.get(key), b.get(key), path + '/' + key)
        return result
    return [] if a == b else [{'路径': path, '取消': a, '保留': b}]

def metrics(d, n):
    assert n > 50, '共同有效窗口不足1秒，不能使用当前姿态统计'
    vx = d['base_linear_velocity_body_array'][:n, 0].astype(float)
    target = d['commanded_velocity_array'][:n, 0].astype(float)
    angle = (d['base_orientation_array'][50:n] + np.pi) % (2 * np.pi) - np.pi
    tilt = np.rad2deg(np.arccos(np.clip(np.cos(angle[:, 1]) * np.cos(angle[:, 2]), -1, 1)))
    power = np.abs(d['joint_torques_array'][:n] * d['joint_velocities_array'][:n]).sum(axis=1)
    distance = float(d['distance_increment_array'][:n].sum())
    return {'共同窗口秒': n * .02, '平均前向速度米每秒': float(vx.mean()),
            '前向速度均方根误差米每秒': float(np.sqrt(np.mean((vx - target) ** 2))),
            '横滚标准差度': float(np.rad2deg(angle[:, 2]).std()),
            '俯仰标准差度': float(np.rad2deg(angle[:, 1]).std()),
            '横滚均值度': float(np.rad2deg(angle[:, 2]).mean()),
            '俯仰均值度': float(np.rad2deg(angle[:, 1]).mean()),
            '倾角95分位度': float(np.percentile(tilt, 95)),
            '最大倾角度': float(tilt.max()),
            '每米绝对关节机械功焦耳': float(power.sum() * .02 / distance) if distance > 1e-9 else None}

def parse(job, protocol):
    run = Path(job['目录'])
    condition = json.loads((run / 'gpt-评估条件.json').read_text())
    p = next(run.glob('eval_*/metrics_summary.json'))
    summary = json.loads(p.read_text())
    source = p.parent / 'plots/sim_data.npz'
    d = np.load(source)
    assert condition['权重SHA256'] == protocol['权重SHA256']
    assert Path(summary['used_checkpoint_path']).resolve() == Path(protocol['权重路径']).resolve()
    assert summary['random_sim_steps'] == 0 and summary['total_sim_steps'] == 500
    for k in ('base_position_array', 'base_orientation_array', 'joint_torques_array', 'joint_velocities_array', 'base_linear_velocity_body_array'):
        assert len(d[k]) == 500 and np.isfinite(d[k]).all()
    np.testing.assert_allclose(d['commanded_velocity_array'], np.tile([condition['指令速度米每秒'], 0, 0], (500, 1)), atol=1e-6)
    audit = np.load(run / 'gpt-实际策略输入.npz')
    assert audit['实际输入'].shape == (500, 236) and audit['策略动作'].shape == (500, 12)
    np.testing.assert_array_equal(audit['原始观测'], audit['实际输入'])
    assert np.isfinite(audit['实际输入']).all() and np.isfinite(audit['策略动作']).all()
    end = min(summary['automatic_reset_steps'], default=500)
    records = json.loads((run / 'gpt-重置前真实状态.json').read_text()) if (run / 'gpt-重置前真实状态.json').exists() else []
    reasons = sorted({k for r in records for k, v in r['终止项'].items() if v and k != 'time_out'})
    early = bool(end < 499 or reasons)
    pos = d['base_position_array'][:end]
    assert len(pos) > 0
    progress = float(condition['起点'][0] - pos[-1, 0])
    lateral = float(abs(pos[-1, 1] - condition['起点'][1]))
    stair_step = stairs.classify(d, summary)[0] if job['场景'] == '8.5厘米楼梯' else None
    passed = stair_step is not None if job['场景'] == '8.5厘米楼梯' else (not early and progress >= condition['指令速度米每秒'] * 10 * .7 and lateral <= 1)
    route_reasons = []
    if not passed:
        if early:
            route_reasons.append('提前终止')
        if job['场景'] == '8.5厘米楼梯':
            route_reasons.append('10秒内未满足四足越过平台条件')
        else:
            if progress < condition['指令速度米每秒'] * 10 * .7:
                route_reasons.append('前进不足目标距离70%')
            if lateral > 1:
                route_reasons.append('侧向偏移超过1米')
    return {**job, '路线通过': bool(passed), '无提前终止': not early, '提前终止原因': reasons,
            '路线未通过原因': route_reasons,
            '有效步数': int(end), '前进米': progress, '侧向偏移米': lateral,
            '楼梯通过秒': (stair_step + 1) * .02 if stair_step is not None else None,
            '轨迹': str(source), '轨迹SHA256': sha(source), '地形SHA256': sha(run / 'gpt-实际地形.npz')}

def audit_pair(a, b):
    ra, rb = Path(a['目录']), Path(b['目录'])
    diffs = difference(cfg(ra / 'gpt-构造环境配置.yaml'), cfg(rb / 'gpt-构造环境配置.yaml'))
    assert {x['路径'] for x in diffs} == {'/events/randomize_com/params/com_range/' + axis for axis in 'xyz'}, diffs
    for x in diffs:
        assert x['取消'] == [0., 0.] and x['保留'] == [-.03, .03]
    da, db = [json.loads((r / 'gpt-运行时随机化.json').read_text()) for r in (ra, rb)]
    for k in ('刚体名称', '各刚体质量千克', '材料属性', '构造后随机数摘要'):
        assert da[k] == db[k], (a['种子'], a['场景'], k)
    index = da['刚体名称'].index('base')
    ca, cb = np.array(da['局部质心']), np.array(db['局部质心'])
    delta = cb[index, :3] - ca[index, :3]
    assert np.max(np.abs(delta)) <= .030001 and np.linalg.norm(delta) > 0
    cb[index, :3] = ca[index, :3]
    np.testing.assert_array_equal(ca, cb)
    ia, ib = [json.loads((r / 'gpt-初始状态.json').read_text()) for r in (ra, rb)]
    assert ia == ib, ('初始状态不一致', a['种子'], a['场景'])
    with np.load(ra / 'gpt-实际地形.npz') as x, np.load(rb / 'gpt-实际地形.npz') as y:
        for k in x.files:
            np.testing.assert_array_equal(x[k], y[k])
    end = min(a['有效步数'], b['有效步数'])
    with np.load(ra / 'gpt-实际策略输入.npz') as x, np.load(rb / 'gpt-实际策略输入.npz') as y:
        np.testing.assert_array_equal(x['原始观测'][0], y['原始观测'][0])
        np.testing.assert_array_equal(x['策略动作'][0], y['策略动作'][0])
        np.testing.assert_array_equal(x['随机数摘要'][:end], y['随机数摘要'][:end])
    for row in (a, b):
        with np.load(row['轨迹']) as d:
            row['共同窗口指标'] = metrics(d, end)
    return {'种子': a['种子'], '场景': a['场景'], '核验': '通过', '配置差异': diffs,
            '共同窗口秒': end * .02, '实际质心偏移厘米': (delta * 100).tolist(),
            '质量材料地形初始状态首步观测一致': True, '共同窗口逐步随机数一致': True}

def aggregate(rows):
    lookup = {(r['条件'], r['种子'], r['场景']): r for r in rows}
    seeds = list(range(40, 50))
    rng = np.random.default_rng(20261026)
    resamples = rng.integers(0, 10, (10000, 10))
    keys = list(rows[0]['共同窗口指标'])
    result = []
    for scene in (*SCENES, '四场景等权平均'):
        scenes = SCENES if scene == '四场景等权平均' else (scene,)
        item = {'场景': scene, '条件统计': {}, '配对差值_保留减取消': {}}
        for condition in CONDITIONS:
            group = [lookup[condition, seed, s] for seed in seeds for s in scenes]
            item['条件统计'][condition] = {'回合数': len(group), '路线通过数': sum(r['路线通过'] for r in group),
                '无提前终止数': sum(r['无提前终止'] for r in group),
                '十种子全部所选场景通过数': sum(all(lookup[condition, seed, s]['路线通过'] for s in scenes) for seed in seeds)}
        for key in keys:
            values = {c: np.array([np.mean([lookup[c, seed, s]['共同窗口指标'][key] for s in scenes]) for seed in seeds]) for c in CONDITIONS}
            for c in CONDITIONS:
                item['条件统计'][c][key] = {'均值': float(values[c].mean()), '种子间标准差': float(values[c].std(ddof=1))}
            delta = values[CONDITIONS[1]] - values[CONDITIONS[0]]
            item['配对差值_保留减取消'][key] = {'均值': float(delta.mean()), '配对自助法95%区间': np.percentile(delta[resamples].mean(axis=1), [2.5, 97.5]).tolist()}
        for flag in ('路线通过', '无提前终止'):
            delta = np.array([np.mean([int(lookup[CONDITIONS[1], seed, s][flag]) - int(lookup[CONDITIONS[0], seed, s][flag]) for s in scenes]) for seed in seeds])
            item['配对差值_保留减取消'][flag] = {'比例差': float(delta.mean()), '配对自助法95%区间': np.percentile(delta[resamples].mean(axis=1), [2.5, 97.5]).tolist()}
        result.append(item)
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--partial', action='store_true')
    args = p.parse_args()
    protocol = json.loads((OUT / 'gpt-预定评估协议.json').read_text())
    assert sha(protocol['权重路径']) == protocol['权重SHA256']
    jobs = json.loads((OUT / 'gpt-运行清单.json').read_text())
    rows = [parse(j, protocol) for j in jobs if (Path(j['目录']) / 'gpt-评估完成.json').exists()]
    lookup = {(r['种子'], r['场景'], r['条件']): r for r in rows}
    audits = []
    for seed in range(40, 50):
        for scene in SCENES:
            if all((seed, scene, c) in lookup for c in CONDITIONS):
                audits.append(audit_pair(*(lookup[seed, scene, c] for c in CONDITIONS)))
    write('gpt-配对条件核验.json', {'已完成回合': len(rows), '已核验配对': len(audits), '逐对核验': audits})
    print('已完成', len(rows), '回合，已核验', len(audits), '对，全部通过', flush=True)
    if args.partial:
        return
    assert len(rows) == 80 and len(audits) == 40, '尚未完成全部测试'
    for scene in SCENES:
        assert len({r['地形SHA256'] for r in rows if r['场景'] == scene}) == 1
    summary = aggregate(rows)
    write('gpt-逐回合结果.json', rows)
    write('gpt-配对统计汇总.json', summary)
    flat = [{**{k: v for k, v in r.items() if k != '共同窗口指标'}, **r['共同窗口指标']} for r in rows]
    with (OUT / 'gpt-80回合逐次结果.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
