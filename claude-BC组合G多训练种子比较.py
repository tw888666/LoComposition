"""G 组（w=0.012，mix_alpha=0.2）三个训练种子的比较与汇总。

主比较：G 相对 F（同训练种子，只改力矩平方混合比例 0.1→0.2）。
另与 C、E（同训练种子）以及训练种子 46 的 A、B 比较；达标线与 F 组相同。
指标、窗口、统计、步态口径全部复用 claude-BC组合F多训练种子比较.py。
"""
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('claude_f_compare', ROOT/'claude-BC组合F多训练种子比较.py')
fc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fc)
single, metrics, ce = fc.single, fc.metrics, fc.ce
SEEDS, SCENES, DISPLAY, KEYS, STAIRS = fc.SEEDS, fc.SCENES, fc.DISPLAY, fc.KEYS, fc.STAIRS
TRAIN_SEEDS = (46, 47, 48)
OUT = ROOT/'claude-BC组合G多训练种子-20261109'
RES = OUT/'比较结果'
G_COEF = 'r_G = -0.012 × (0.8 × Σ|τ·q̇| + 0.2 × s × Στ²)，s = 0.17440421040278947'
ORDER = {l: i for i, l in enumerate('ABCEFG')}
read = fc.read


def source(train_seed, label):
    if label == 'G':
        return OUT/f'训练种子{train_seed}', OUT/'正式任务'/f'G-训练种子{train_seed}'/'claude-状态.json'
    return fc.source(train_seed, label)


def load():
    groups = {46: ('A', 'B', 'C', 'E', 'F', 'G'), 47: ('C', 'E', 'F', 'G'), 48: ('C', 'E', 'F', 'G')}
    old_hashes = {(r['组别'], r['种子'], r['场景']): r['轨迹SHA256'] for r in read(single.OLD/'评估/gpt-逐次评估指标.json')}
    pretrain = read(single.OLD/'gpt-固定参考尺度.json')['checkpoint_sha256']
    lookup, missing, audit = {}, [], []
    for train_seed, labels in groups.items():
        for label in labels:
            root, status_path = source(train_seed, label)
            model_dir = None
            if label in ('F', 'G') or train_seed != 46:
                state = read(status_path)
                assert state['状态'] == '训练评估完成' and state['已完成迭代'] == 5000 and state['评估完成场数'] == 18, status_path
                prefix = 'claude' if label in ('F', 'G') else 'gpt'
                assert read(status_path.parent/f'{prefix}-启动核验.json')['状态'] == '通过'
                model_dir = Path(state['模型目录'])
                assert metrics.sha(model_dir/'model_5000.pt') == state['最终权重SHA256'], (train_seed, label)
                training = read(model_dir/'gpt-四组微调配置.json')
                assert training['随机种子'] == train_seed and training['实验组'] == label and training['追加迭代'] == 5000
                assert training['来源SHA256'] == pretrain
                audit.append({'训练种子': train_seed, '组别': label, '模型目录': str(model_dir), '最终权重SHA256': state['最终权重SHA256']})
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
                    assert record['轨迹SHA256'] == old_hashes[label, evaluation_seed, scene]
                lookup[train_seed, label, evaluation_seed, scene] = (record, data, cfg)
    if missing:
        raise RuntimeError(f'缺少{len(missing)}场正式评估：{missing[:3]}')
    checks = []
    for evaluation_seed, scene in itertools.product(SEEDS, SCENES):
        keys = [k for k in lookup if k[2:] == (evaluation_seed, scene)]
        reference = lookup[keys[0]]
        for key in keys:
            record, _, cfg = lookup[key]
            for field in ('地形SHA256', '电机SHA256', '环境配置SHA256', '动力学'):
                assert record[field] == reference[0][field], (key, field)
            for field in ('起点', '时长秒', '指令速度米每秒', '控制周期秒', '地形种子', '电机', '高程图', '归一化', '动作', '随机化'):
                assert cfg[field] == reference[2][field], (key, field)
            np.testing.assert_allclose(record['首步观测'], reference[0]['首步观测'], rtol=0, atol=1e-6)
        checks.append({'评估种子': evaluation_seed, '场景': scene, '模型数': len(keys), '协议': '全部一致'})
    return lookup, checks, audit


def failures(lookup, train_seed, label):
    out = []
    for s, sc in itertools.product(SEEDS, SCENES):
        r = lookup[train_seed, label, s, sc][0]
        if not r['成功']:
            o = r['自身窗口指标']
            out.append({'评估种子': s, '场景': sc, '提前终止': r['提前终止'], '末帧侧偏米': round(o['末帧侧偏米'], 2),
                        '前向净位移米': round(o['前向净位移米'], 2), '航向均方根度': round(o['航向偏差均方根度'], 1)})
    return out


def main():
    RES.mkdir(exist_ok=True)
    lookup, checks, audit = load()
    labels46 = ('A', 'B', 'C', 'E', 'F', 'G')
    rows46 = fc.seed_rows(lookup, 46, labels46)
    seed46 = {l: single.summarize(rows46, l) for l in labels46}
    stairs46 = {l: single.summarize([r for r in rows46 if r['场景'] == STAIRS], l) for l in labels46}
    pairs46 = {f'G/{b}': single.paired(fc.sub(lookup, 46, ('G', b)), 'G', b) for b in ('A', 'B', 'C', 'F')}

    per_seed, all_rows = [], []
    cmp = ('C', 'E', 'F', 'G')
    for ts in TRAIN_SEEDS:
        rows = fc.seed_rows(lookup, ts, cmp); all_rows.extend(rows)
        per_seed.append({'训练种子': ts, **{l: single.summarize(rows, l) for l in cmp},
                         '楼梯': {l: single.summarize([r for r in rows if r['场景'] == STAIRS], l) for l in cmp},
                         '逐场景': {sc: {l: single.summarize([r for r in rows if r['场景'] == sc], l) for l in cmp} for sc in SCENES}})
    g_vs = {b: fc.cross_seed(per_seed, 'G', b) for b in ('F', 'C', 'E')}

    b_energy = fc.own_summary(lookup, 46, 'B')['每米有效前进机械功焦耳']
    criteria = []
    for label in ('F', 'G'):
        for ts in TRAIN_SEEDS:
            whole, stairs = fc.own_summary(lookup, ts, label), fc.own_summary(lookup, ts, label, (STAIRS,))
            items = {'能耗': (whole['每米有效前进机械功焦耳'], whole['每米有效前进机械功焦耳'] <= 1.05*b_energy),
                     '楼梯航向': (stairs['航向偏差均方根度'], stairs['航向偏差均方根度'] <= 6.4),
                     '楼梯末帧侧偏': (stairs['最大末帧横向偏移米'], stairs['最大末帧横向偏移米'] <= 1.0),
                     '通过': (whole['成功次数'], whole['成功次数'] >= 17)}
            criteria.append({'组别': label, '训练种子': ts, '逐项': {k: {'实测': v[0], '达标': bool(v[1])} for k, v in items.items()},
                             '全部达标': all(v[1] for v in items.values()), '失败场次': failures(lookup, ts, label)})

    gait_rows, gait = fc.gait_table(lookup)
    result = {'能耗项': G_COEF, '模型审计': audit, '环境配对核验组数': len(checks), '训练种子46六组共同窗口': seed46,
              '训练种子46楼梯': stairs46, '训练种子46配对': {k: v['逐指标变化'] for k, v in pairs46.items()},
              '逐训练种子CEFG': per_seed, '跨训练种子G相对': g_vs, '达标线': criteria, 'B每米机械功': b_energy, '楼梯步态': gait}
    metrics.save(RES/'claude-G组比较汇总.json', result)
    metrics.write_csv(RES/'claude-逐训练种子CEFG总体.csv', [dict(训练种子=s['训练种子'], **s[l]) for s in per_seed for l in cmp])
    metrics.write_csv(RES/'claude-CEFG配对逐次指标.csv', [{k: v for k, v in r.items() if not isinstance(v, dict)} for r in all_rows])
    metrics.write_csv(RES/'claude-楼梯步态汇总.csv', [{k: v for k, v in g.items() if not isinstance(v, list)} for g in gait])

    f, p = fc.fmt, fc.pct
    L = ['# G 组（w=0.012，mix_alpha=0.2）多训练种子比较结果', '',
         f'G 组能耗项：{G_COEF}。与 F 组（mix_alpha=0.1）只差力矩平方混合比例，其余协议相同。'
         f'共 {len(lookup)} 条正式轨迹，{len(checks)} 组评估条件下所有模型的地形、电机、环境、动力学与首步观测逐项一致。', '',
         '## 1. 达标线（各自完整首次回合）', '',
         f'门槛：每米机械功 ≤ {1.05*b_energy:.2f}（B 的 1.05 倍）；楼梯航向均方根 ≤ 6.4°；楼梯末帧侧偏最大值 ≤ 1 米；原任务通过 ≥ 17/18。', '',
         '| 组别 | 训练种子 | 每米机械功 | 楼梯航向均方根 | 楼梯末帧侧偏 | 原任务通过 | 全部达标 | 失败场次 |',
         '|---|---|---:|---:|---:|---:|---|---|']
    for c in criteria:
        it = c['逐项']; m = lambda k: '✅' if it[k]['达标'] else '❌'
        fl = '；'.join(f"{x['场景']}/评估{x['评估种子']}（侧偏 {x['末帧侧偏米']} 米）" for x in c['失败场次']) or '无'
        L.append(f"| {c['组别']} | {c['训练种子']} | {it['能耗']['实测']:.2f} {m('能耗')} | {it['楼梯航向']['实测']:.2f} {m('楼梯航向')} | "
                 f"{it['楼梯末帧侧偏']['实测']:.2f} {m('楼梯末帧侧偏')} | {it['通过']['实测']}/18 {m('通过')} | {'是' if c['全部达标'] else '否'} | {fl} |")
    L += ['', '## 2. 三个训练种子：G 与 F、C、E 同训练种子比较', '', '每个训练种子内 C/E/F/G 取共同首次回合窗口。', '',
          '| 训练种子 | 组别 | 原任务通过 | 航向均方根 | 最大侧偏 | 初始方向速度误差 | 机械功/有效米 | 力矩平方积分/有效米 | 楼梯航向均方根 |',
          '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in per_seed:
        for l in ('E', 'C', 'F', 'G'):
            r = s[l]
            L.append(f"| {s['训练种子']} | {l} | {r['成功次数']}/18 | " + ' | '.join(f(r[DISPLAY[i][0]]) for i in (0, 1, 4, 5, 6)) +
                     f" | {f(s['楼梯'][l]['航向偏差均方根度'])} |")
    L += ['', '跨三个训练种子（先种子内平均，再种子间平均；括号内为 G 优于对照的种子数）：', '',
          '| 指标 | F 均值 | G 均值 | G 相对 F | G 相对 C | G 相对 E |', '|---|---:|---:|---:|---:|---:|']
    for key, title in DISPLAY:
        a = g_vs['F'][key]
        cells = [f"{p(g_vs[b][key]['变化百分比'])}（{g_vs[b][key]['改善种子数'] if g_vs[b][key]['改善种子数'] is not None else '—'}/3）" for b in ('F', 'C', 'E')]
        L.append(f"| {title} | {f(a['对照组均值'])} | {f(a['实验组均值'])} | " + ' | '.join(cells) + ' |')
    L += ['', '平地 1.0 与崎岖 1.0（F 组失败集中的两个场景）航向均方根，度：', '',
          '| 训练种子 | 场景 | C | E | F | G |', '|---|---|---:|---:|---:|---:|']
    for s in per_seed:
        for sc in ('平地1.0', '崎岖1.0'):
            L.append(f"| {s['训练种子']} | {sc} | " + ' | '.join(f(s['逐场景'][sc][l]['航向偏差均方根度'], 2) for l in ('C', 'E', 'F', 'G')) + ' |')
    L += ['', '## 3. 训练种子 46：六组比较', '', '| 组别 | 原任务通过 | 航向均方根 | 最大侧偏 | 有效前进 | 初始方向速度误差 | 机械功/有效米 | 力矩平方积分/有效米 | 楼梯航向均方根 |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for l in labels46:
        r = seed46[l]
        L.append(f"| {l} | {r['成功次数']}/18 | " + ' | '.join(f(r[DISPLAY[i][0]]) for i in (0, 1, 2, 4, 5, 6)) + f" | {f(stairs46[l]['航向偏差均方根度'])} |")
    L += ['', 'G 相对各组的配对变化（训练种子 46，区间按评估种子重采样；* 为区间不跨 0）：', '',
          '| 指标 | G 相对 A | G 相对 B | G 相对 C | G 相对 F |', '|---|---:|---:|---:|---:|']
    for key, title in DISPLAY:
        cells = []
        for b in ('A', 'B', 'C', 'F'):
            ch = pairs46[f'G/{b}']['逐指标变化'][key]; ci = ch['配对种子描述性95区间绝对差']
            cells.append(p(ch['变化百分比']) + ('*' if ci is not None and (ci[0] > 0 or ci[1] < 0) else ''))
        L.append(f'| {title} | ' + ' | '.join(cells) + ' |')
    L += ['', '## 4. 楼梯爬升段步态（口径同 F 组报告）', '',
          '| 训练种子 | 组别 | FL+RR 偏航（°/s） | FR+RL 偏航（°/s） | 正负抵消 | 过 x=-11 累计航向（°） | 末帧航向（3 次，°） | 机械功率（瓦） | 力矩平方和 |',
          '|---|---|---:|---:|---|---:|---|---:|---:|']
    for g in sorted(gait, key=lambda g: (g['训练种子'], ORDER[g['组别']])):
        L.append(f"| {g['训练种子']} | {g['组别']} | {g['FL+RR支撑偏航角速度度每秒']:+.2f} | {g['FR+RL支撑偏航角速度度每秒']:+.2f} | "
                 f"{'是' if g['两对角相偏航正负抵消'] else '否'} | {g['过x=-11时累计航向变化度']:+.1f} | {g['逐评估种子末帧累计航向变化度']} | "
                 f"{g['机械功率瓦']:.1f} | {g['力矩平方和牛米平方']:.0f} |")
    L += ['', '## 5. 口径与限制', '', '训练种子内对六场景×三个评估种子等权平均，再对训练种子等权平均；区间只按训练种子整组重采样，仅作描述。',
          '三次微调共享同一个预训练检查点；评估只有一套固定地形布局；A、B 只有训练种子 46。能耗指标是关节机械功代理。', '']
    (RES/'claude-G组多训练种子比较报告.md').write_text('\n'.join(L))
    metrics.save(RES/'claude-比较完成.json', {'状态': '完成', '正式轨迹': len(lookup), '实现SHA256': metrics.sha(Path(__file__))})
    print(f'完成：{len(lookup)} 条轨迹')


if __name__ == '__main__':
    main()

