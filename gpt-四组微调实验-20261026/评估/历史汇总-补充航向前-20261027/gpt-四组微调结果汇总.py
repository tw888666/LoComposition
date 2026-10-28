"""审计90场统一评估，按配对种子和共同首次回合窗口汇总能耗代理与任务表现。"""
from pathlib import Path
import argparse
import csv
from datetime import datetime
import hashlib
import json
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parent
LABELS = ('A0', 'A', 'B', 'C', 'D')
SEEDS = (46, 47, 48)
SCENES = ('平地0.4', '平地0.8', '平地1.0', '崎岖1.0', '楼梯8.5厘米', '坡道0.8')
KEYS = ('前向速度均方根误差米每秒', '平均绝对机械功率瓦', '平均力矩平方和牛米平方',
        '每米绝对机械功焦耳', '每米力矩平方积分牛米平方秒每米')


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metrics(data, n, cfg):
    """只使用首次回合、物理推进后且重置前的真实帧，首帧距离从实际起点计算。"""
    dt = float(cfg['控制周期秒'])
    if n <= 0:
        raise ValueError('没有可比较的真实数据。')
    tau = data['关节力矩'][:n].astype(np.float64)
    qdot = data['关节角速度'][:n].astype(np.float64)
    pos = data['机身位置'][:n].astype(np.float64)
    velocity = data['机身线速度机体系'][:n].astype(np.float64)
    command = data['指令速度'][:n].astype(np.float64)
    angular = data['机身角速度机体系'][:n].astype(np.float64)
    power = np.abs(tau * qdot).sum(axis=1)
    squared = np.square(tau).sum(axis=1)
    distance_steps = np.linalg.norm(np.diff(np.vstack([np.asarray(cfg['起点'])[:2], pos[:, :2]]), axis=0), axis=1)
    distance = float(distance_steps.sum())
    energy = float(power.sum() * dt)
    q_integral = float(squared.sum() * dt)
    pitch_roll = data['机身姿态偏航俯仰横滚'][:n, 1:].astype(np.float64)
    tilt = np.arccos(np.clip(np.prod(np.cos(pitch_roll), axis=1), -1, 1))
    return {'有效步数': n, '窗口秒': n * dt,
            '平均前向速度米每秒': float(velocity[:, 0].mean()),
            '前向速度均方根误差米每秒': float(np.sqrt(np.mean(np.square(velocity[:, 0] - command[:, 0])))),
            '侧向速度均方根误差米每秒': float(np.sqrt(np.mean(np.square(velocity[:, 1] - command[:, 1])))),
            '偏航角速度均方根误差弧度每秒': float(np.sqrt(np.mean(np.square(angular[:, 2] - command[:, 2])))),
            '平均绝对机械功率瓦': float(power.mean()), '绝对机械功焦耳': energy,
            '平均力矩平方和牛米平方': float(squared.mean()), '力矩平方积分牛米平方秒': q_integral,
            '水平路径长度米': distance, '前向净位移米': float(cfg['起点'][0] - pos[-1, 0]),
            '末帧侧偏米': float(abs(pos[-1, 1] - cfg['起点'][1])),
            '最大单步水平位移米': float(distance_steps.max()),
            '每米绝对机械功焦耳': energy / distance if distance > 1e-9 else None,
            '每米力矩平方积分牛米平方秒每米': q_integral / distance if distance > 1e-9 else None,
            '机身总倾角均方根度': float(np.rad2deg(np.sqrt(np.mean(np.square(tilt))))),
            '约束违规步百分比': {key.removeprefix('约束__'): float(np.mean(np.any(data[key][:n].reshape(n, -1) > 0, axis=1)) * 100)
                          for key in data if key.startswith('约束__')}}


def continuous_gate(gate, window=10):
    if len(gate) < window:
        return None
    found = np.flatnonzero(np.convolve(gate.astype(int), np.ones(window, dtype=int), 'valid') == window)
    return int(found[0] + window - 1) if len(found) else None


def read_run(run):
    marker = json.loads((run / 'gpt-评估完成.json').read_text())
    assert marker['状态'] == '完成' and marker['正式评估'] and marker['归一化冻结核验']
    cfg = json.loads((run / 'gpt-评估条件.json').read_text())
    assert cfg['正式评估'] and cfg['地形种子'] == 46
    assert sha(cfg['权重']) == cfg['权重SHA256'], '评估完成后权重内容发生变化。'
    source = run / 'gpt-控制步真实数据.npz'
    with np.load(source) as archived:
        data = {key: archived[key] for key in archived.files}
    planned = int(round(cfg['时长秒'] / cfg['控制周期秒']))
    assert len(data['关节力矩']) == planned == marker['控制步数']
    for key, array in data.items():
        assert len(array) == planned and np.isfinite(array).all(), (run, key)
    assert data['关节力矩'].shape == data['关节角速度'].shape == (planned, 12)
    np.testing.assert_allclose(data['指令速度'], np.tile([cfg['指令速度米每秒'], 0., 0.], (planned, 1)), rtol=0, atol=1e-6)
    with np.load(run / 'gpt-实际策略输入.npz') as audit:
        assert audit['原始观测'].shape == (planned, 236)
        assert audit['策略动作'].shape == (planned, 12)
        np.testing.assert_array_equal(audit['原始观测'], audit['实际输入'])
        first = audit['原始观测'][0].copy()
        assert np.isfinite(audit['策略动作']).all()
    resets = np.flatnonzero(data['终止'] | data['超时'])
    n = int(resets[0] + 1) if len(resets) else planned
    early = bool(np.any(data['终止'][:n]))
    own = metrics(data, n, cfg)
    scene = cfg['场景']
    gate_step = None
    if scene in ('楼梯8.5厘米', '坡道0.8'):
        feet, pos = data['足端世界坐标'][:n], data['机身位置'][:n]
        angles = data['机身姿态偏航俯仰横滚'][:n]
        upright = np.cos(angles[:, 1]) * np.cos(angles[:, 2]) >= np.cos(np.pi / 6)
        if scene == '楼梯8.5厘米':
            gate = (np.all(feet[:, :, 0] <= -11.05, axis=1) & np.all(feet[:, :, 2] >= -.03, axis=1)
                    & np.all((feet[:, :, 1] >= 13.5) & (feet[:, :, 1] <= 18.5), axis=1) & (pos[:, 2] >= .2) & upright)
        else:
            gate = (np.all(feet[:, :, 0] <= -6.7, axis=1) & np.all(feet[:, :, 2] >= .205, axis=1)
                    & np.all((feet[:, :, 1] >= -17.) & (feet[:, :, 1] <= -15.), axis=1) & (pos[:, 2] >= .435) & upright)
        gate_step = continuous_gate(gate)
        passed = gate_step is not None
    else:
        passed = (not early and n == planned and own['前向净位移米'] >= cfg['指令速度米每秒'] * cfg['时长秒'] * .7
                  and own['末帧侧偏米'] <= 1.)
    record = {'组别': cfg['组别'], '检查点': cfg['检查点'], '种子': cfg['种子'], '场景': scene,
              '成功': bool(passed), '无提前终止完成': not early and n == planned,
              '首次回合有效步数': n, '提前终止': early,
              '越过平台秒': (gate_step + 1) * cfg['控制周期秒'] if gate_step is not None else None,
              '自身窗口指标': own, '目录': str(run), '轨迹': str(source), '轨迹SHA256': sha(source),
              '权重SHA256': cfg['权重SHA256'], '地形SHA256': sha(run / 'gpt-实际地形.npz'),
              '环境配置SHA256': sha(run / 'gpt-构造环境配置.yaml'),
              '电机SHA256': sha(run / 'gpt-运行时电机配置.json'),
              '动力学': json.loads((run / 'gpt-运行时随机化.json').read_text()), '首步观测': first.tolist()}
    return record, data, cfg


def make_plot(out, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams['axes.unicode_minus'] = False
    specs = [('成功率', '任务成功率'), (KEYS[0], '前向速度误差（米/秒）'), (KEYS[1], '绝对机械功率（瓦）'),
             (KEYS[2], '力矩平方和（牛米平方）'), (KEYS[3], '单位距离机械功（焦耳/米）'),
             (KEYS[4], '单位距离力矩平方积分\n（牛米平方秒/米）')]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    colors = ['#687786', '#3973ac', '#dc9136', '#3f8d6d', '#966fa8']
    for ax, (key, title) in zip(axes.flat, specs):
        vals = [s[key] for s in summaries]
        ax.bar(LABELS, vals, color=colors)
        if key != '成功率':
            intervals = np.asarray([s[key + '95区间'] for s in summaries])
            ax.errorbar(LABELS, vals, yerr=[np.maximum(0, np.asarray(vals) - intervals[:, 0]),
                                          np.maximum(0, intervals[:, 1] - vals)], fmt='none', ecolor='#222222', capsize=3)
        else:
            ax.set_ylim(0, 1.1)
        ax.set_title(title, fontproperties=font)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('四组微调与原30000次基线：6场景 × 3个配对评估种子', fontproperties=font, fontsize=15)
    fig.text(.5, .01, '连续指标使用每个场景和种子的五策略共同首次回合窗口；误差线为按3个配对种子重采样的95%区间。',
             ha='center', fontproperties=font, fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .95))
    fig.savefig(out / 'gpt-四组微调统一指标对比.png', dpi=180)
    plt.close(fig)


def mean_comparison_judgements(summaries):
    """只按成功率、速度误差及两类单位距离代价判断本批均值是否同时不差。"""
    lines = []
    for summary in summaries:
        if summary['组别'] not in ('C', 'D'):
            continue
        for reference in ('A', 'B'):
            comparison = summary['配对对照比较'][reference]
            changes = [
                ('成功率', comparison['成功率差百分点'], True, '个百分点'),
                ('前向速度误差', comparison[KEYS[0]]['绝对差'], False, '米/秒'),
                ('每米机械功', comparison[KEYS[3]]['绝对差'], False, '焦耳/米'),
                ('每米力矩平方积分', comparison[KEYS[4]]['绝对差'], False, '牛米平方秒/米'),
            ]
            directions, details = [], []
            for name, change, higher_is_better, unit in changes:
                benefit = change if higher_is_better else -change
                direction = 0 if abs(benefit) <= 1e-12 else (1 if benefit > 0 else -1)
                directions.append(direction)
                verdict = '改善' if direction > 0 else '退化' if direction < 0 else '持平'
                details.append(f'{name}{verdict}（{change:+.4f}{unit}）')
            if min(directions) >= 0 and max(directions) > 0:
                verdict = '本批均值有优势：四项均不差，且至少一项改善'
            elif min(directions) < 0 and max(directions) > 0:
                verdict = '存在指标折中，不能称为四项整体优势'
            elif min(directions) < 0:
                verdict = '本批均值未见优势，存在退化指标'
            else:
                verdict = '四项均值持平，未见改善'
            lines.append(f"- {summary['组别']} 对 {reference}：{verdict}。" + '；'.join(details) + '。')
    return lines


def summarize(out):
    out.mkdir(parents=True, exist_ok=True)
    selected, missing = [], []
    for label in LABELS:
        options = sorted((out / label).glob('检查点*'), key=lambda x: int(x.name.removeprefix('检查点')))
        checkpoint_dir = options[-1] if options else None
        for seed in SEEDS:
            for scene in SCENES:
                run = checkpoint_dir / f'种子{seed}' / scene if checkpoint_dir else None
                if run is not None and (run / 'gpt-评估完成.json').exists():
                    selected.append(run)
                else:
                    missing.append({'组别': label, '种子': seed, '场景': scene})
    save(out / 'gpt-汇总进度.json', {'状态': '数据齐全待审计' if not missing else '评估尚未完成',
         '正式评估完成场数': len(selected), '计划场数': 90, '缺失': missing})
    if missing:
        print(json.dumps({'已完成': len(selected), '计划': 90, '状态': '不生成最终结论，等待全部正式评估。'}, ensure_ascii=False))
        raise SystemExit(2)
    parsed = [read_run(path) for path in selected]
    lookup = {(r['组别'], r['种子'], r['场景']): (r, data, cfg) for r, data, cfg in parsed}
    checks = []
    for seed in SEEDS:
        for scene in SCENES:
            pairs = [lookup[(label, seed, scene)] for label in LABELS]
            ref = pairs[0][0]
            n = min(r['首次回合有效步数'] for r, _, _ in pairs)
            observation_errors = {}
            for record, data, cfg in pairs:
                for field in ('地形SHA256', '电机SHA256', '环境配置SHA256', '动力学'):
                    assert record[field] == ref[field], (seed, scene, record['组别'], field)
                observation_errors[record['组别']] = float(np.max(np.abs(np.asarray(record['首步观测']) - ref['首步观测'])))
                np.testing.assert_allclose(record['首步观测'], ref['首步观测'], rtol=0, atol=1e-6)
                record['共同窗口指标'] = metrics(data, n, cfg)
            checks.append({'种子': seed, '场景': scene, '共同步数': n,
                           '共同秒数': n * pairs[0][2]['控制周期秒'], '环境地形电机动力学首观测配对': '通过',
                           '首观测最大绝对差': observation_errors, '首观测绝对容差': 1e-6,
                           '其他配置与动力学': '严格一致'})
    rows = [r for r, _, _ in parsed]
    rng = np.random.default_rng(20261026)
    resamples = rng.integers(0, len(SEEDS), size=(10000, len(SEEDS)))
    summaries, by_scene, boot = [], [], {}
    for label in LABELS:
        subset = [r for r in rows if r['组别'] == label]
        summary = {'组别': label, '检查点': subset[0]['检查点'], '成功次数': sum(r['成功'] for r in subset),
                   '评估次数': len(subset), '无提前终止完成次数': sum(r['无提前终止完成'] for r in subset)}
        summary['成功率'] = summary['成功次数'] / len(subset)
        for key in KEYS:
            values = np.array([[lookup[(label, seed, scene)][0]['共同窗口指标'][key] for scene in SCENES] for seed in SEEDS], dtype=float)
            assert np.isfinite(values).all(), (label, key, '单位距离指标出现不可用值，需人工检查原始距离。')
            sampled = values[resamples].mean(axis=(1, 2))
            boot[(label, key)] = sampled
            summary[key] = float(values.mean())
            summary[key + '95区间'] = np.quantile(sampled, [.025, .975]).tolist()
        summary['平均约束违规步百分比'] = {term: float(np.mean([r['共同窗口指标']['约束违规步百分比'][term] for r in subset]))
                                        for term in subset[0]['共同窗口指标']['约束违规步百分比']}
        summaries.append(summary)
        for scene in SCENES:
            scene_rows = [r for r in subset if r['场景'] == scene]
            by_scene.append({'组别': label, '场景': scene, '成功次数': sum(r['成功'] for r in scene_rows), '评估次数': len(scene_rows),
                             **{key: float(np.mean([r['共同窗口指标'][key] for r in scene_rows])) for key in KEYS}})
    for summary in summaries:
        label = summary['组别']
        summary['配对对照比较'] = {}
        for reference in ('A0', 'A', 'B'):
            baseline = next(s for s in summaries if s['组别'] == reference)
            comparison = {'成功率差百分点': (summary['成功率'] - baseline['成功率']) * 100}
            for key in KEYS:
                relative = (boot[(label, key)] / boot[(reference, key)] - 1) * 100
                difference = boot[(label, key)] - boot[(reference, key)]
                comparison[key] = {'变化百分比': (summary[key] / baseline[key] - 1) * 100,
                    '配对种子95区间百分比': np.quantile(relative, [.025, .975]).tolist(),
                    '绝对差': summary[key] - baseline[key], '配对种子95区间绝对差': np.quantile(difference, [.025, .975]).tolist()}
            summary['配对对照比较'][reference] = comparison
        summary['相对原30000次基线变化'] = summary['配对对照比较']['A0']
    save(out / 'gpt-逐次评估指标.json', rows)
    save(out / 'gpt-配对核验.json', checks)
    save(out / 'gpt-统一评估汇总.json', {'状态': '90场全部完成且配对核验通过', '组别汇总': summaries, '逐场景汇总': by_scene})
    flat = []
    for r in rows:
        item = {k: r[k] for k in ('组别', '检查点', '种子', '场景', '成功', '无提前终止完成', '提前终止', '越过平台秒', '轨迹')}
        for prefix, block in (('自身窗口_', r['自身窗口指标']), ('共同窗口_', r['共同窗口指标'])):
            item.update({prefix + k: v for k, v in block.items() if not isinstance(v, dict)})
        flat.append(item)
    write_csv(out / 'gpt-逐次评估指标.csv', flat)
    write_csv(out / 'gpt-逐场景汇总.csv', by_scene)
    write_csv(out / 'gpt-五组总体指标.csv', [{k: v for k, v in s.items() if not isinstance(v, dict)} for s in summaries])
    comparisons = []
    for s in summaries:
        for reference, delta in s['配对对照比较'].items():
            if s['组别'] == reference:
                continue
            comparisons.append({'组别': s['组别'], '对照组': reference, '成功率差百分点': delta['成功率差百分点'],
                **{key + '_变化百分比': delta[key]['变化百分比'] for key in KEYS},
                **{key + '_配对95区间百分比': delta[key]['配对种子95区间百分比'] for key in KEYS}})
    write_csv(out / 'gpt-相对对照组配对变化.csv', comparisons)
    make_plot(out, summaries)
    lines = ['# 四组微调统一评估结果', '', '90场正式评估全部完成：原30000次基线A0与A/B/C/D，各6个场景、3个配对评估种子（46、47、48）。', '',
             '| 组别 | 成功次数 | 前向速度误差（米/秒） | 机械功率（瓦） | 力矩平方和（牛米平方） | 机械功/距离（焦耳/米） | 力矩平方积分/距离（牛米平方秒/米） |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['组别']} | {s['成功次数']}/18 | " + ' | '.join(f'{s[key]:.4f}' for key in KEYS) + ' |')
    lines += ['', '主要比较采用C/D相对继续原奖励微调的A组及加强原功率惩罚的B组；A0是未微调的原权重参考。下表负值表示相应连续指标下降，需同时考虑成功率和速度误差。', '',
              '| 组别/对照 | 成功率差（百分点） | 速度误差变化 | 每米机械功变化及配对95%区间 | 每米力矩平方积分变化及配对95%区间 |',
              '|---|---:|---:|---:|---:|']
    for s in summaries:
        if s['组别'] not in ('C', 'D'):
            continue
        for reference in ('A', 'B'):
            delta = s['配对对照比较'][reference]
            cells = []
            for key in (KEYS[3], KEYS[4]):
                item = delta[key]
                lo, hi = item['配对种子95区间百分比']
                cells.append(f"{item['变化百分比']:+.2f}% [{lo:+.2f}%, {hi:+.2f}%]")
            lines.append(f"| {s['组别']}/{reference} | {delta['成功率差百分点']:+.2f} | {delta[KEYS[0]]['变化百分比']:+.2f}% | " + ' | '.join(cells) + ' |')
    lines += ['', '连续指标按照每个场景和种子中五个策略的最短首次回合窗口计算，并按场景和种子等权平均。另存每个策略自身完整首次回合的功率、能量、力矩平方积分、真实水平路径长度、净位移和终止情况。所有物理数据在自动重置前采集，包含真实终态，不把重置位移计入行走距离。', '',
              '平地和崎岖成功要求：完整运行且无提前终止，前进净位移达到指令距离70%，末帧侧偏不超过1米。楼梯成功沿用四足跨过平台、直立连续10步的规则。坡道使用原固定地形的约6.8度上坡：四足越过世界X=-6.7米，足端高度不低于0.205米、机身高度不低于0.435米，直立连续10步。越过平台之后的失败另行保留，不撤销已完成的越障。', '',
              '平地、崎岖和楼梯各10秒；坡道5秒，以避免驶入相邻随机地形。坡道指令为0.8米/秒；平地分别为0.4、0.8、1.0米/秒；崎岖和楼梯为1.0米/秒。各组使用训练Go2电机配置、真实高程图、冻结的各自观测归一化统计和确定性策略动作。地形、初始动力学、首步原始观测及电机配置已逐场景配对核验。', '',
              '机械功率为逐关节力矩乘角速度的绝对值之和；机械功为它按控制周期的积分。力矩平方和及其积分仅作为电机铜损代理，未乘电阻/力矩常数，也未标定驱动器、传动和电池损耗。旧30k评估数据与本次训练电机配置不同；如固定归一化尺度来自旧评估，它们只用于奖励量级匹配，不能当作物理标定。', '',
              '95%区间按3个配对评估种子整组重采样，每组包含全部6个场景。只有一次微调训练种子，且只有3个动力学/观测评估种子、1个固定地形布局；区间只用于这批仿真的描述，不能代替多训练种子或真实机器人验证。', '',
              '约束违规统计保存在汇总JSON（结构化数据文件）中：逐项报告任一关节/足端违反该限制的控制步比例，采用重置前的原始约束值，不能把概率终止信号等同于实际跌倒。', '',
              '基于本批均值的判断：', '']
    lines += mean_comparison_judgements(summaries)
    lines += ['', '上述判断只使用单个训练种子的跨场景均值，不表示统计显著，也不能保证每个场景都改善。机械功与力矩平方积分是仿真能耗代理；这些结果不能证明真实电能节约。', '',
              '当前任务的统一评估与统计已完成。下一步：结合成功率、速度误差和两类能耗代理筛选候选策略；随后按需要扩展训练种子与地形布局。', '']
    (out / 'gpt-四组微调统一评估报告.md').write_text('\n'.join(lines))
    save(out / 'gpt-汇总完成.json', {'状态': '完成', '正式评估场数': 90, '配对核验组数': len(checks)})
    print(json.dumps({'状态': '完成', '组别汇总': summaries}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'gpt-四组微调实验-20261026')
    args = parser.parse_args()
    out = args.output_root.resolve() / '评估'
    try:
        summarize(out)
    except (Exception, SystemExit) as error:
        if isinstance(error, SystemExit) and error.code in (None, 0):
            raise
        out.mkdir(parents=True, exist_ok=True)
        save(out / 'gpt-汇总失败.json', {
            '状态': '汇总失败', '时间': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
            '异常类型': type(error).__name__, '错误': str(error),
            '说明': '缺少正式评估或审计、统计、报告生成失败；不得据此声明实验全部完成。',
        })
        raise
    (out / 'gpt-汇总失败.json').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
