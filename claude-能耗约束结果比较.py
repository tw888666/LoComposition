"""能耗拉格朗日约束组 L 与 A/B/C/E 的比较；复用四组微调的原始指标实现与配对窗口口径。"""
import argparse
from datetime import datetime
import importlib.util
import itertools
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parent
V1_DIR = ROOT / 'claude-能耗约束实验-20261029'
V2_DIR = ROOT / 'claude-能耗约束实验-v2-20261101'
FOUR_GROUP = ROOT / 'gpt-四组微调实验-20261026'
BASELINES = {'A': FOUR_GROUP, 'B': FOUR_GROUP, 'C': FOUR_GROUP, 'E': ROOT / 'gpt-纯机械功率消融-20261028'}
spec = importlib.util.spec_from_file_location('gpt_original_metrics', ROOT / 'gpt-四组微调结果汇总.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
spec = importlib.util.spec_from_file_location('gpt_ablation_compare', ROOT / 'gpt-纯机械功率消融比较.py')
ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation)
DISPLAY = ablation.DISPLAY
# 默认复现第一版的三个检查点；--subject 可改为任意"标签=实验目录:检查点"
DEFAULT_SUBJECTS = [f'L{c}={V1_DIR}:{c}' for c in (5000, 3149, 1099)]


def parse_subject(text):
    label, _, rest = text.partition('=')
    root, _, checkpoint = rest.rpartition(':')
    assert label and root and checkpoint.isdigit(), f'--subject 格式应为 标签=实验目录:检查点，收到 {text!r}'
    return label, Path(root), int(checkpoint)


def run_dir(root, eval_label, checkpoint, seed, scene):
    return root / '评估' / eval_label / f'检查点{checkpoint}' / f'种子{seed}' / scene


def load(subjects):
    groups = [(label, root, label, 5000) for label, root in BASELINES.items()]
    groups += [(label, root, 'L', checkpoint) for label, root, checkpoint in subjects]
    assert len({g[0] for g in groups}) == len(groups), '组别标签重复'
    lookup, missing = {}, []
    for (label, root, eval_label, checkpoint), seed, scene in itertools.product(groups, original.SEEDS, original.SCENES):
        run = run_dir(root, eval_label, checkpoint, seed, scene)
        if not (run / 'gpt-评估完成.json').is_file():
            missing.append(str(run))
            continue
        record, data, cfg = original.read_run(run)
        assert (record['种子'], record['场景'], record['检查点']) == (seed, scene, checkpoint)
        record = dict(record, 组别=label)
        lookup[label, seed, scene] = (record, data, cfg)
    if missing:
        raise RuntimeError(f'缺少{len(missing)}场正式评估：{missing[:3]}')
    labels = [g[0] for g in groups]
    checks = []
    for seed, scene in itertools.product(original.SEEDS, original.SCENES):
        reference = lookup['A', seed, scene]
        for label in labels:
            record, _, cfg = lookup[label, seed, scene]
            for key in ('地形SHA256', '电机SHA256', '动力学'):
                assert record[key] == reference[0][key], (label, seed, scene, key)
            for key in ('起点', '时长秒', '指令速度米每秒', '控制周期秒', '地形种子', '电机', '高程图', '归一化', '动作', '随机化'):
                assert cfg[key] == reference[2][key], (label, seed, scene, key)
            np.testing.assert_allclose(record['首步观测'], reference[0]['首步观测'], rtol=0, atol=1e-6)
        checks.append({'种子': seed, '场景': scene, '组别': labels, '地形电机动力学与评估条件': '一致'})
    return lookup, labels, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=V1_DIR, help='结果写入 <out-dir>/比较结果')
    parser.add_argument('--subject', action='append', help='标签=实验目录:检查点；可重复，默认第一版三个检查点')
    args = parser.parse_args()
    subjects = [parse_subject(s) for s in (args.subject or DEFAULT_SUBJECTS)]
    lookup, labels, checks = load(subjects)
    out = args.out_dir.resolve() / '比较结果'
    out.mkdir(parents=True, exist_ok=True)
    # 各组相对同一对照的两两共同窗口，避免某组提前终止截短其他组
    comparisons = {f'{s}/{b}': ablation.paired(lookup, s, b)
                   for s, _, _ in subjects for b in ('A', 'C', 'E')}
    for name, comparison in comparisons.items():
        comparison.pop('逐次指标')
    own = {label: {'原任务通过': sum(lookup[label, s, sc][0]['成功'] for s, sc in itertools.product(original.SEEDS, original.SCENES)),
                   '无提前终止完成': sum(lookup[label, s, sc][0]['无提前终止完成'] for s, sc in itertools.product(original.SEEDS, original.SCENES)),
                   **{key: ablation.mean([lookup[label, s, sc][0]['自身窗口指标'][key]
                                          for s, sc in itertools.product(original.SEEDS, original.SCENES)])
                      for key in original.ALL_KEYS}}
           for label in labels}
    original.save(out / 'claude-能耗约束比较汇总.json', {
        '更新时间': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), '组别': labels,
        '各组自身完整首次回合均值': own, '配对比较': comparisons})
    original.save(out / 'claude-能耗约束配对核验.json', checks)
    original.write_csv(out / 'claude-全部配对比较.csv', [dict(比较=name, 指标=key, **values)
                                                for name, comparison in comparisons.items()
                                                for key, values in comparison['逐指标变化'].items()])
    print(json.dumps({'自身窗口': {k: {m: v[m] for m in ('原任务通过', '无提前终止完成', '每米有效前进机械功焦耳',
                                                         '沿初始方向速度均方根误差米每秒', '航向偏差均方根度', '平均绝对机械功率瓦')}
                                for k, v in own.items()},
                      '配对': {name: {'成功次数差': c['成功次数差'],
                                      **{label: c['逐指标变化'][key]['变化百分比'] for key, label in DISPLAY[:7]}}
                               for name, c in comparisons.items()}}, ensure_ascii=False, indent=1), flush=True)


if __name__ == '__main__':
    main()
