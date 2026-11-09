"""F 组（B 的权重 0.012 + C 的 10% 力矩平方）三个训练种子的比较与汇总。

1. 训练种子 46：F 与已有的 A、B、C、E（均为训练种子 46）做五组共同窗口比较和逐对配对比较。
2. 训练种子 46/47/48：F 与 C、E 做同训练种子比较，按训练种子整组统计（沿用 gpt-CE多训练种子比较.py 口径）。
3. 三条达标线检查，以及楼梯爬升段的偏航、对角支撑相偏航角速度分析。
指标、窗口和统计函数全部复用 gpt-纯机械功率消融比较.py / gpt-CE多训练种子比较.py，不重新定义。
"""
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'claude-BC组合F多训练种子-20261105'
RES = OUT/'比较结果'


def _module(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT/file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ce = _module('claude_ce_multi', 'gpt-CE多训练种子比较.py')
single = ce.single
metrics = ce.metrics
CE_OUT = ce.OUT
SEEDS, SCENES, DISPLAY, KEYS = metrics.SEEDS, metrics.SCENES, single.DISPLAY, single.KEYS
TRAIN_SEEDS = (46, 47, 48)
STAIRS = '楼梯8.5厘米'
F_COEF = 'r_F = -0.012 × (0.9 × Σ|τ·q̇| + 0.1 × s × Στ²)，s = 0.17440421040278947'


def read(path):
    return json.loads(Path(path).read_text())


def source(train_seed, label):
    """返回 (评估根目录, 状态文件或 None, 模型目录或 None)。"""
    if label == 'F':
        status = OUT/'正式任务'/f'F-训练种子{train_seed}'/'claude-状态.json'
        return OUT/f'训练种子{train_seed}', status
    if train_seed == 46:
        if label == 'E':
            return single.OUT, single.OUT/'gpt-总状态.json'
        return single.OLD, None
    return CE_OUT/f'训练种子{train_seed}', CE_OUT/'正式任务'/f'{label}-训练种子{train_seed}'/'gpt-状态.json'


def load():
    groups = {46: ('A', 'B', 'C', 'E', 'F'), 47: ('C', 'E', 'F'), 48: ('C', 'E', 'F')}
    old_hashes = {(r['组别'], r['种子'], r['场景']): r['轨迹SHA256']
                  for r in read(single.OLD/'评估/gpt-逐次评估指标.json')}
    pretrain = read(single.OLD/'gpt-固定参考尺度.json')['checkpoint_sha256']
    lookup, missing, audit = {}, [], []
    for train_seed, labels in groups.items():
        for label in labels:
            root, status_path = source(train_seed, label)
            model_dir = None
            if label == 'F' or train_seed != 46:
                state = read(status_path)
                assert state['状态'] == '训练评估完成' and state['已完成迭代'] == 5000 and state['评估完成场数'] == 18, status_path
                prefix = 'claude' if label == 'F' else 'gpt'
                assert read(status_path.parent/f'{prefix}-启动核验.json')['状态'] == '通过'
                model_dir = Path(state['模型目录'])
                assert metrics.sha(model_dir/'model_5000.pt') == state['最终权重SHA256'], (train_seed, label)
                training = read(model_dir/'gpt-四组微调配置.json')
                assert training['随机种子'] == train_seed and training['实验组'] == label and training['追加迭代'] == 5000
                assert training['来源SHA256'] == pretrain
                audit.append({'训练种子': train_seed, '组别': label, '模型目录': str(model_dir),
                              '最终权重SHA256': state['最终权重SHA256'], '启动核验': '通过'})
            for evaluation_seed, scene in itertools.product(SEEDS, SCENES):
                run = root/'评估'/label/'检查点5000'/f'种子{evaluation_seed}'/scene
                if not (run/'gpt-评估完成.json').is_file():
                    missing.append(str(run)); continue
                record, data, cfg = metrics.read_run(run)
                assert (record['组别'], record['种子'], record['场景'], record['检查点']) == (label, evaluation_seed, scene, 5000)
                assert cfg['控制周期秒'] == .02 and cfg.get('初始航向弧度', np.pi) == np.pi
                if model_dir is not None:
                    assert Path(cfg['权重']).resolve() == (model_dir/'model_5000.pt').resolve()
                elif label != 'E':
                    assert record['轨迹SHA256'] == old_hashes[label, evaluation_seed, scene], '历史轨迹已变化'
                lookup[train_seed, label, evaluation_seed, scene] = (record, data, cfg)
    if missing:
        raise RuntimeError(f'缺少{len(missing)}场正式评估，拒绝生成结论：{missing[:3]}')
    # 同一评估种子×场景下，所有模型的地形、电机、环境、动力学与首步观测必须一致。
    checks = []
    for evaluation_seed, scene in itertools.product(SEEDS, SCENES):
        keys = [k for k in lookup if k[2:] == (evaluation_seed, scene)]
        reference = lookup[keys[0]]
        errors = {}
        for key in keys:
            record, _, cfg = lookup[key]
            for field in ('地形SHA256', '电机SHA256', '环境配置SHA256', '动力学'):
                assert record[field] == reference[0][field], (key, field)
            for field in ('起点', '时长秒', '指令速度米每秒', '控制周期秒', '地形种子', '电机', '高程图', '归一化', '动作', '随机化'):
                assert cfg[field] == reference[2][field], (key, field)
            np.testing.assert_allclose(record['首步观测'], reference[0]['首步观测'], rtol=0, atol=1e-6)
            errors[f'{key[1]}/{key[0]}'] = float(np.max(np.abs(np.array(record['首步观测'])-reference[0]['首步观测'])))
        checks.append({'评估种子': evaluation_seed, '场景': scene, '模型数': len(keys),
                       '地形电机环境动力学协议': '全部一致', '首原始观测最大绝对差': errors})
    return lookup, checks, audit


def sub(lookup, train_seed, labels):
    """转成 single.* 函数需要的 (组别, 评估种子, 场景) 键。"""
    return {(label, s, sc): lookup[train_seed, label, s, sc] for label, s, sc in itertools.product(labels, SEEDS, SCENES)}


def seed_rows(lookup, train_seed, labels):
    rows = single.window_rows(sub(lookup, train_seed, labels), labels)
    for row in rows:
        row['训练种子'] = train_seed
    return rows


def cross_seed(per_seed, subject, baseline):
    stats = {}
    for key in KEYS:
        r = ce.training_statistics([s[subject][key] for s in per_seed], [s[baseline][key] for s in per_seed],
                                   key == '沿初始方向有效前进距离米')
        stats[key] = {'实验组': subject, '对照组': baseline, **{k.replace('C均值', '实验组均值').replace('E均值', '对照组均值'): v for k, v in r.items()}}
    for field in ('成功次数', '无提前终止完成次数'):
        r = ce.training_statistics([s[subject][field] for s in per_seed], [s[baseline][field] for s in per_seed], True)
        stats[field] = {k.replace('C均值', '实验组均值').replace('E均值', '对照组均值'): v for k, v in r.items()}
    return stats


def own_summary(lookup, train_seed, label, scenes=SCENES):
    """各自完整首次回合（不截到共同窗口），用于达标线检查。"""
    recs = [lookup[train_seed, label, s, sc][0] for s, sc in itertools.product(SEEDS, scenes)]
    own = [r['自身窗口指标'] for r in recs]
    out = {'场数': len(recs), '成功次数': sum(r['成功'] for r in recs),
           '无提前终止完成次数': sum(r['无提前终止完成'] for r in recs)}
    for key in ('每米有效前进机械功焦耳', '每米有效前进力矩平方积分牛米平方秒每米', '航向偏差均方根度',
                '最大绝对横向偏移米', '终点绝对横向偏移米', '终点有符号航向偏差度', '沿初始方向有效前进距离米'):
        out[key] = single.mean([o[key] for o in own])
    out['最大末帧横向偏移米'] = max(o['终点绝对横向偏移米'] for o in own)
    out['逐评估种子末帧航向度'] = [o['终点有符号航向偏差度'] for o in own]
    return out


TH = 5.0  # 足端接触力阈值（牛），超过视为支撑；与 11-05 B 组步态分析一致


def stairs_gait(data, n):
    """楼梯爬升段（机身 x 从 -8.5 到 -11.05）的对角支撑相偏航角速度等，与 11-05 B 组分析同口径。"""
    contact = data['足端接触力'][:n] > TH  # 顺序 FL FR RL RR
    yaw_rate = np.degrees(data['机身角速度机体系'][:n, 2])
    tau, dq = data['关节力矩'][:n], data['关节角速度'][:n]
    power = np.abs(tau*dq)
    x = data['机身位置'][:n, 0]
    yaw = np.degrees(np.unwrap(data['机身姿态偏航俯仰横滚'][:n, 0])); yaw -= yaw[0]
    mask = np.ones(n, bool); mask[:50] = False
    mask &= (x <= -8.5) & (x >= -11.05)
    d1 = mask & contact[:, 0] & contact[:, 3] & ~contact[:, 1] & ~contact[:, 2]  # FL+RR 支撑
    d2 = mask & contact[:, 1] & contact[:, 2] & ~contact[:, 0] & ~contact[:, 3]  # FR+RL 支撑
    passed = np.flatnonzero(x <= -11)
    nan_mean = lambda v: float(v.mean()) if len(v) else float('nan')
    return {'FL+RR支撑偏航角速度度每秒': nan_mean(yaw_rate[d1]), 'FR+RL支撑偏航角速度度每秒': nan_mean(yaw_rate[d2]),
            '爬升段平均偏航角速度度每秒': nan_mean(yaw_rate[mask]),
            '过x=-11时累计航向变化度': float(yaw[passed[0]]) if len(passed) else float('nan'),
            '末帧累计航向变化度': float(yaw[-1]), '末帧x米': float(x[-1]),
            '升高米': float(data['机身位置'][n-1, 2]-data['机身位置'][0, 2]),
            '机械功率瓦': float(power.sum(1).mean()), '力矩平方和牛米平方': float((tau**2).sum(1).mean()),
            '大腿关节功率瓦': float(power[:, 4:8].sum(1).mean()), '小腿关节功率瓦': float(power[:, 8:12].sum(1).mean())}


def gait_table(lookup):
    rows = []
    for (train_seed, label, evaluation_seed, scene), (record, data, cfg) in sorted(lookup.items()):
        if scene != STAIRS:
            continue
        rows.append({'训练种子': train_seed, '组别': label, '评估种子': evaluation_seed,
                     **stairs_gait(data, record['首次回合有效步数'])})
    groups = {}
    for r in rows:
        groups.setdefault((r['训练种子'], r['组别']), []).append(r)
    summary = []
    for (train_seed, label), items in groups.items():
        entry = {'训练种子': train_seed, '组别': label}
        for key in items[0]:
            if key in ('训练种子', '组别', '评估种子'):
                continue
            entry[key] = float(np.nanmean([i[key] for i in items]))
        entry['逐评估种子末帧累计航向变化度'] = [round(i['末帧累计航向变化度'], 1) for i in items]
        a, b = entry['FL+RR支撑偏航角速度度每秒'], entry['FR+RL支撑偏航角速度度每秒']
        entry['两对角相偏航正负抵消'] = bool(a*b < 0)
        summary.append(entry)
    return rows, summary


def plot(path, per_seed, seed46):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    colors = {'A': '#8c8c8c', 'B': '#d08a3a', 'C': '#3f8d6d', 'E': '#966fa8', 'F': '#c0392b'}
    x = np.arange(len(per_seed))
    for ax, (key, title) in zip(axes.flat, [DISPLAY[i] for i in (0, 1, 2, 4, 5, 6)]):
        for i, label in enumerate(('E', 'C', 'F')):
            values = [np.nan if r[label][key] is None else r[label][key] for r in per_seed]
            ax.bar(x+(i-1)*.26, values, .24, label=label, color=colors[label])
        for label, style in (('A', ':'), ('B', '--')):
            value = seed46[label][key]
            if value is not None:
                ax.axhline(value, color=colors[label], ls=style, lw=1.4, label=f'{label}（训练种子46）')
        ax.set_xticks(x, [str(r['训练种子']) for r in per_seed])
        ax.set_xlabel('微调训练种子', fontproperties=font)
        ax.set_title(title, fontproperties=font, fontsize=11)
        ax.grid(axis='y', alpha=.2)
        ax.legend(prop=font, fontsize=8)
    fig.suptitle('F 组（B 权重 + C 力矩平方混合）三个训练种子与 C/E 对比：每柱为该模型 18 场评估均值\n'
                 '同一训练种子内 C/E/F 取共同首次回合窗口；虚线/点线为训练种子 46 的 B/A（五组共同窗口）',
                 fontproperties=font, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .91]); fig.savefig(path, dpi=160); plt.close(fig)


def fmt(v, digits=4):
    return '不可定义' if v is None else f'{v:.{digits}f}'


def pct(v):
    return '不可定义' if v is None else f'{v:+.2f}%'


def main():
    RES.mkdir(exist_ok=True)
    lookup, checks, audit = load()

    # 1) 训练种子 46：五组共同窗口 + F 对各组配对
    labels46 = ('A', 'B', 'C', 'E', 'F')
    rows46 = seed_rows(lookup, 46, labels46)
    seed46 = {label: single.summarize(rows46, label) for label in labels46}
    stairs46 = {label: single.summarize([r for r in rows46 if r['场景'] == STAIRS], label) for label in labels46}
    pairs46 = {f'F/{b}': single.paired(sub(lookup, 46, ('F', b)), 'F', b) for b in ('A', 'B', 'C', 'E')}

    # 2) 三个训练种子：C/E/F 共同窗口，按训练种子整组统计
    per_seed, all_rows = [], []
    for train_seed in TRAIN_SEEDS:
        rows = seed_rows(lookup, train_seed, ('C', 'E', 'F'))
        all_rows.extend(rows)
        per_seed.append({'训练种子': train_seed, **{label: single.summarize(rows, label) for label in ('C', 'E', 'F')},
                         '楼梯': {label: single.summarize([r for r in rows if r['场景'] == STAIRS], label) for label in ('C', 'E', 'F')}})
    f_vs_c = cross_seed(per_seed, 'F', 'C')
    f_vs_e = cross_seed(per_seed, 'F', 'E')
    stairs_f_vs_c = cross_seed([{'训练种子': s['训练种子'], **s['楼梯']} for s in per_seed], 'F', 'C')

    # 3) 达标线（各自完整首次回合）
    b_energy = own_summary(lookup, 46, 'B')['每米有效前进机械功焦耳']
    a_stairs_heading = own_summary(lookup, 46, 'A', (STAIRS,))['航向偏差均方根度']
    criteria = []
    for train_seed in TRAIN_SEEDS:
        whole, stairs = own_summary(lookup, train_seed, 'F'), own_summary(lookup, train_seed, 'F', (STAIRS,))
        items = {
            '每米机械功 ≤ 1.05×B': (whole['每米有效前进机械功焦耳'], 1.05*b_energy, whole['每米有效前进机械功焦耳'] <= 1.05*b_energy),
            '楼梯航向均方根 ≤ 6.4 度': (stairs['航向偏差均方根度'], 6.4, stairs['航向偏差均方根度'] <= 6.4),
            '楼梯末帧侧偏（3 次中最大）≤ 1 米': (stairs['最大末帧横向偏移米'], 1.0, stairs['最大末帧横向偏移米'] <= 1.0),
            '原任务通过 ≥ 17/18': (whole['成功次数'], 17, whole['成功次数'] >= 17),
        }
        criteria.append({'训练种子': train_seed, '逐项': {k: {'实测': v[0], '门槛': v[1], '达标': bool(v[2])} for k, v in items.items()},
                         '全部达标': all(v[2] for v in items.values()), '全程': whole, '楼梯': stairs})
    reference = {'B每米有效前进机械功（训练种子46，自身窗口）': b_energy, 'A楼梯航向均方根（训练种子46，自身窗口）': a_stairs_heading}

    gait_rows, gait = gait_table(lookup)

    result = {'状态': '数据齐全且环境配对核验通过', '能耗项': F_COEF, '模型审计': audit,
              '训练种子46五组共同窗口': seed46, '训练种子46五组楼梯共同窗口': stairs46,
              '训练种子46配对': {k: {kk: v[kk] for kk in ('实验组汇总', '对照组汇总', '成功次数差', '无提前终止完成次数差', '逐指标变化')}
                            for k, v in pairs46.items()},
              '逐训练种子CEF共同窗口': per_seed, '跨训练种子F相对C': f_vs_c, '跨训练种子F相对E': f_vs_e,
              '跨训练种子楼梯F相对C': stairs_f_vs_c, '达标线参考': reference, '达标线检查': criteria,
              '楼梯步态分析': gait,
              '统计口径': '训练种子内对六场景×三个评估种子等权平均，再对训练种子等权平均；区间只按训练种子整组重采样（27 种有放回组合），仅作描述。'}
    metrics.save(RES/'claude-F组比较汇总.json', result)
    metrics.save(RES/'claude-F组环境配对核验.json', checks)
    metrics.write_csv(RES/'claude-训练种子46五组总体.csv', [seed46[l] for l in labels46])
    metrics.write_csv(RES/'claude-逐训练种子CEF总体.csv', [dict(训练种子=s['训练种子'], **s[l]) for s in per_seed for l in ('C', 'E', 'F')])
    metrics.write_csv(RES/'claude-CEF配对逐次指标.csv', [{k: v for k, v in r.items() if not isinstance(v, dict)} for r in all_rows + rows46])
    metrics.write_csv(RES/'claude-楼梯步态逐次.csv', gait_rows)
    metrics.write_csv(RES/'claude-楼梯步态汇总.csv', [{k: v for k, v in g.items() if not isinstance(v, list)} for g in gait])

    # ---------- 报告 ----------
    L = ['# F 组（B+C 组合）多训练种子比较结果', '',
         f'F 组能耗项：{F_COEF}。从同一个 30000 次预训练模型微调 5000 次，训练种子 46/47/48，各 18 场正式评估（6 场景 × 3 评估种子）。',
         '训练与评估使用冻结实现（相对 C/E 复验冻结实现只加了分支 F）。所有模型在同一评估种子×场景下的地形、电机、环境配置、动力学和首步观测逐项核验一致。', '',
         '## 1. 达标线检查', '',
         f'参考值：B 组（训练种子 46）每米有效前进机械功 {b_energy:.2f} 焦耳，1.05 倍为 {1.05*b_energy:.2f}；A 组楼梯航向均方根 {a_stairs_heading:.2f} 度。'
         '本节用各模型自身完整首次回合。', '',
         '| 训练种子 | 每米机械功（≤ %.2f） | 楼梯航向均方根（≤ 6.4°） | 楼梯末帧侧偏最大值（≤ 1 米） | 原任务通过（≥ 17/18） | 全部达标 |' % (1.05*b_energy),
         '|---|---:|---:|---:|---:|---|']
    for c in criteria:
        it = c['逐项']; mark = lambda k: '✅' if it[k]['达标'] else '❌'
        keys = list(it)
        L.append(f"| {c['训练种子']} | {it[keys[0]]['实测']:.2f} {mark(keys[0])} | {it[keys[1]]['实测']:.2f} {mark(keys[1])} | "
                 f"{it[keys[2]]['实测']:.2f} {mark(keys[2])} | {it[keys[3]]['实测']}/18 {mark(keys[3])} | {'是' if c['全部达标'] else '否'} |")
    L += ['', '## 2. 训练种子 46：A/B/C/E/F 五组比较', '', '五组取共同首次回合窗口。', '',
          '| 组别 | 原任务通过 | 航向均方根（度） | 最大侧偏（米） | 有效前进（米） | 初始方向速度误差（米/秒） | 机械功（焦耳/有效米） | 力矩平方积分/有效米 |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label in labels46:
        r = seed46[label]
        L.append(f"| {label} | {r['成功次数']}/18 | " + ' | '.join(fmt(r[DISPLAY[i][0]]) for i in (0, 1, 2, 4, 5, 6)) + ' |')
    L += ['', '楼梯 8.5 厘米场景（3 次评估均值，五组共同窗口）：', '',
          '| 组别 | 通过 | 航向均方根（度） | 最大侧偏（米） | 机械功（焦耳/有效米） | 力矩平方积分/有效米 |', '|---|---:|---:|---:|---:|---:|']
    for label in labels46:
        r = stairs46[label]
        L.append(f"| {label} | {r['成功次数']}/3 | " + ' | '.join(fmt(r[DISPLAY[i][0]]) for i in (0, 1, 5, 6)) + ' |')
    L += ['', 'F 相对各组的配对变化（每对单独取共同窗口；区间按评估种子重采样，仅作描述）：', '',
          '| 指标 | F 相对 A | F 相对 B | F 相对 C | F 相对 E |', '|---|---:|---:|---:|---:|']
    for key, title in DISPLAY:
        cells = []
        for b in ('A', 'B', 'C', 'E'):
            ch = pairs46[f'F/{b}']['逐指标变化'][key]
            ci = ch['配对种子描述性95区间绝对差']
            tag = '' if ci is None else ('*' if ci[0] > 0 or ci[1] < 0 else '')
            cells.append(pct(ch['变化百分比']) + tag)
        L.append(f'| {title} | ' + ' | '.join(cells) + ' |')
    L += ['', '\\* 表示描述性 95% 区间不跨 0。', '',
          '## 3. 三个训练种子：F 与 C、E 同训练种子比较', '', '每个训练种子内 C/E/F 取共同首次回合窗口。', '',
          '| 训练种子 | 组别 | 原任务通过 | 航向均方根 | 最大侧偏 | 初始方向速度误差 | 机械功/有效米 | 力矩平方积分/有效米 | 楼梯航向均方根 |',
          '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in per_seed:
        for label in ('E', 'C', 'F'):
            r = s[label]
            L.append(f"| {s['训练种子']} | {label} | {r['成功次数']}/18 | " +
                     ' | '.join(fmt(r[DISPLAY[i][0]]) for i in (0, 1, 4, 5, 6)) + f" | {fmt(s['楼梯'][label]['航向偏差均方根度'])} |")
    L += ['', '单位：度、米、米/秒、焦耳/有效米、(牛·米)²·秒/有效米、度。', '',
          '跨三个训练种子（先种子内平均，再种子间平均）：', '',
          '| 指标 | E 均值 | C 均值 | F 均值 | F 相对 C | F 优于 C 的种子数 | F 相对 E | F 优于 E 的种子数 |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key, title in DISPLAY:
        a, b = f_vs_c[key], f_vs_e[key]
        L.append(f"| {title} | {fmt(b['对照组均值'])} | {fmt(a['对照组均值'])} | {fmt(a['实验组均值'])} | {pct(a['变化百分比'])} | "
                 f"{a['改善种子数'] if a['改善种子数'] is not None else '不可定义'}/3 | {pct(b['变化百分比'])} | "
                 f"{b['改善种子数'] if b['改善种子数'] is not None else '不可定义'}/3 |")
    sc = stairs_f_vs_c['航向偏差均方根度']
    L += ['', f"楼梯场景航向均方根 F 相对 C：{pct(sc['变化百分比'])}，F 优于 C 的种子数 {sc['改善种子数']}/3，"
          f"各种子差值 {[round(v, 2) for v in sc['各训练种子绝对差']]} 度。", '',
          '## 4. 楼梯爬升段步态分析', '',
          '口径与 11-05 的 B 组分析相同：机身 x 在 -8.5 到 -11.05 米之间、去掉前 50 步；足端接触力 > 5 牛视为支撑。用各自首次回合。', '',
          '| 训练种子 | 组别 | FL+RR 支撑偏航（°/s） | FR+RL 支撑偏航（°/s） | 正负抵消 | 过 x=-11 累计航向（°） | 末帧航向（3 次，°） | 机械功率（瓦） | 力矩平方和 | 大腿/小腿功率（瓦） |',
          '|---|---|---:|---:|---|---:|---|---:|---:|---:|']
    order = {l: i for i, l in enumerate('ABCEF')}
    for g in sorted(gait, key=lambda g: (g['训练种子'], order[g['组别']])):
        L.append(f"| {g['训练种子']} | {g['组别']} | {g['FL+RR支撑偏航角速度度每秒']:+.2f} | {g['FR+RL支撑偏航角速度度每秒']:+.2f} | "
                 f"{'是' if g['两对角相偏航正负抵消'] else '否'} | {g['过x=-11时累计航向变化度']:+.1f} | {g['逐评估种子末帧累计航向变化度']} | "
                 f"{g['机械功率瓦']:.1f} | {g['力矩平方和牛米平方']:.0f} | {g['大腿关节功率瓦']:.1f}/{g['小腿关节功率瓦']:.1f} |")
    L += ['', '## 5. 统计口径与限制', '', result['统计口径'],
          '三次微调共享同一个预训练检查点，不是三次从零训练；评估只有一套固定地形布局。B 和 A 只有训练种子 46，所以 F 与 B 的比较只能在训练种子 46 上配对，跨种子只能与 C/E 比较。',
          '能耗指标是关节机械功代理，不等于真实电池能耗。', '',
          '![逐训练种子对比](claude-F组逐训练种子对比.png)', '']
    (RES/'claude-F组多训练种子比较报告.md').write_text('\n'.join(L))
    plot(RES/'claude-F组逐训练种子对比.png', per_seed, seed46)
    metrics.save(RES/'claude-比较完成.json', {'状态': '完成', '正式轨迹': len(lookup), '环境配对核验': len(checks),
                                          '实现SHA256': metrics.sha(Path(__file__))})
    print(f'完成：{len(lookup)} 条轨迹，{len(checks)} 组环境配对核验。')


if __name__ == '__main__':
    main()

