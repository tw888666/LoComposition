"""由统一汇总入口调用，补充五策略方向指标，并对B/C作两组配对复评。"""
from datetime import datetime
from pathlib import Path
import csv
import hashlib
import json
from zoneinfo import ZoneInfo

import numpy as np


SCENES = ('平地0.4', '平地0.8', '平地1.0', '崎岖1.0', '楼梯8.5厘米', '坡道0.8')
SEEDS = (46, 47, 48)
DISPLAY = (
    ('航向偏差均方根度', '航向偏差均方根（度）'),
    ('终点绝对航向偏差度', '终点绝对航向偏差（度）'),
    ('最大绝对横向偏移米', '全程最大绝对横向偏移（米）'),
    ('终点绝对横向偏移米', '终点绝对横向偏移（米）'),
    ('沿初始方向有效前进距离米', '有效前进距离（米）'),
    ('有效前进距离占指令距离比例', '有效前进距离／指令距离'),
    ('有效前进距离占路径长度比例', '有效前进距离／实际路径长度'),
    ('前向速度均方根误差米每秒', '机身前向速度误差（米/秒）'),
    ('沿初始方向速度均方根误差米每秒', '沿初始方向速度误差（米/秒）'),
    ('每米绝对机械功焦耳', '按实际路径长度：机械功（焦耳/米）'),
    ('每米有效前进机械功焦耳', '按有效前进距离：机械功（焦耳/米）'),
    ('每米力矩平方积分牛米平方秒每米', '按实际路径长度：力矩平方积分（牛米平方秒/米）'),
    ('每米有效前进力矩平方积分牛米平方秒每米', '按有效前进距离：力矩平方积分（牛米平方秒/米）'),
)


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(out, by_scene):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams['font.family'] = font.get_name()
    plt.rcParams['axes.unicode_minus'] = False
    specs = [DISPLAY[0], DISPLAY[2], DISPLAY[4], DISPLAY[8], DISPLAY[10], DISPLAY[12]]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    x = np.arange(len(SCENES))
    labels = ['平地\n0.4', '平地\n0.8', '平地\n1.0', '崎岖\n1.0', '楼梯\n8.5厘米', '坡道\n0.8']
    for ax, (key, title) in zip(axes.flat, specs):
        for label, shift, color in [('B', -.18, '#dc9136'), ('C', .18, '#3f8d6d')]:
            values = [next(r for r in by_scene if r['组别']==label and r['场景']==scene)[key] for scene in SCENES]
            ax.bar(x+shift, values, width=.34, color=color, label=label)
        ax.set_xticks(x, labels)
        ax.set_title(title, fontsize=11)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.legend()
    fig.suptitle('B与C：方向保持、有效前进距离与能耗代理', fontsize=16)
    fig.text(.5, .012, '每场景3个配对评估种子的均值；B/C共用完整首次回合窗口。坡道5秒，其余场景10秒。\n有效距离为终点位移在初始朝向上的投影；仅一次训练种子，不能据此推断真实电耗。', ha='center', fontsize=10)
    fig.tight_layout(rect=(0,.065,1,.95))
    fig.savefig(out/'gpt-B与C方向保持及有效前进.png', dpi=180)
    plt.close(fig)


def write_report(out, parsed, metrics, direction_keys):
    lookup = {(r['组别'], r['种子'], r['场景']): (r, data, cfg) for r, data, cfg in parsed}
    if len(lookup) != 90:
        raise ValueError('方向复评要求五策略各18场完整原评估。')
    keys = tuple(dict.fromkeys(direction_keys + tuple(k for k, _ in DISPLAY)))
    pairs = []
    for seed in SEEDS:
        for scene in SCENES:
            b, c = lookup[('B', seed, scene)], lookup[('C', seed, scene)]
            for field in ('环境配置SHA256', '地形SHA256', '电机SHA256', '动力学'):
                if b[0][field] != c[0][field]:
                    raise ValueError(f'B/C配对配置不一致：{seed}/{scene}/{field}')
            n = min(b[0]['首次回合有效步数'], c[0]['首次回合有效步数'])
            for record, data, cfg in (b,c):
                result = metrics(data, n, cfg)
                if not result['有效前进能耗可定义']:
                    raise ValueError('有效前进距离非正，必须单独报告失败，不能将单位距离能耗设为零或静默排除。')
                pairs.append({'组别':record['组别'], '场景':scene, '种子':seed,
                    '原任务判据成功':record['成功'], '无提前终止完成':record['无提前终止完成'],
                    '两组共同窗口秒':result['窗口秒'], '原五组共同窗口秒':record['共同窗口指标']['窗口秒'],
                    '初始航向来源':'评估条件明确记录' if '初始航向弧度' in cfg else '原评估固定根四元数wxyz=[0,0,0,1]，对应180度',
                    '轨迹SHA256':record['轨迹SHA256'], '轨迹':record['轨迹'],
                    **result})
    rng = np.random.default_rng(20261027)
    indices = rng.integers(0, len(SEEDS), (10000,len(SEEDS)))
    summary, boot, by_scene = {}, {}, []
    for label in ('B','C'):
        subset = [r for r in pairs if r['组别']==label]
        summary[label] = {'原任务判据成功次数':sum(r['原任务判据成功'] for r in subset), '场次':len(subset),
            '无提前终止完成次数':sum(r['无提前终止完成'] for r in subset),
            '总有效前进距离米':sum(r['沿初始方向有效前进距离米'] for r in subset),
            '总评估窗口秒':sum(r['两组共同窗口秒'] for r in subset)}
        for key in keys:
            matrix = np.array([[next(r for r in subset if r['种子']==seed and r['场景']==scene)[key]
                                for scene in SCENES] for seed in SEEDS], dtype=float)
            if not np.isfinite(matrix).all():
                raise ValueError(f'存在不可用方向指标：{label}/{key}')
            summary[label][key] = float(matrix.mean())
            boot[(label,key)] = matrix[indices].mean(axis=(1,2))
        for scene in SCENES:
            group = [r for r in subset if r['场景']==scene]
            by_scene.append({'组别':label, '场景':scene, '原任务通过数':sum(r['原任务判据成功'] for r in group),
                **{key:float(np.mean([r[key] for r in group])) for key in keys}})
    changes = {}
    for key in keys:
        b,c = summary['B'][key], summary['C'][key]
        denominator = boot[('B',key)]
        relative = (boot[('C',key)]/denominator - 1)*100 if np.all(np.abs(denominator)>1e-12) else None
        changes[key] = {'B':b, 'C':c, 'C减B':c-b, 'C相对B变化百分比':(c/b-1)*100 if abs(b)>1e-12 else None,
            '配对种子差值95区间':np.quantile(boot[('C',key)]-boot[('B',key)],[.025,.975]).tolist(),
            '配对种子变化百分比95区间':np.quantile(relative,[.025,.975]).tolist() if relative is not None else None}
    sensitivity=[]
    for heading in (10.,20.,30.):
        for side in (.5,1.,2.):
            counts={label:sum(r['原任务判据成功'] and r['无提前终止完成']
                and r['最大绝对航向偏差度']<=heading and r['最大绝对横向偏移米']<=side
                for r in pairs if r['组别']==label) for label in ('B','C')}
            sensitivity.append({'全程航向上限度':heading, '全程侧偏上限米':side, 'B满足次数':counts['B'], 'C满足次数':counts['C'], '每组场次':18})
    all_rows=[]
    for record, _, _ in parsed:
        row={k:record[k] for k in ('组别','种子','场景','成功','提前终止','轨迹SHA256')}
        for prefix in ('自身窗口','共同窗口'):
            block=record[prefix+'指标']
            row[prefix+'_窗口秒']=block['窗口秒']
            row.update({prefix+'_'+key:block[key] for key in direction_keys})
        all_rows.append(row)
    definitions={
        '航向偏差':'每步偏航角减固定初始目标航向，用正弦与余弦的双参数反正切映射至[-180,180]度；报告均方根、最大绝对值和终点绝对值。',
        '横向偏移':'(位置-初始位置)在初始朝向水平垂线上的投影；报告均方根、最大绝对值和终点绝对值。',
        '有效前进距离':'终点水平位移在初始朝向单位向量上的投影；有符号净进展，侧向运动不增加它，后退抵消前进，不累计正向增量。',
        '每米有效前进能耗':'首次回合窗口内总机械功或力矩平方积分除以有效前进距离；非正距离记不可定义并中止自动均值，不能计为零能耗。',
        '沿初始方向速度误差':'将机体系三维线速度按偏航、俯仰、横滚姿态旋转至世界系，再投影至初始方向，与固定前向指令计算均方根误差。',
        '窗口':'五策略总表保留五组共同首次回合窗口；B/C复评单独使用两组共同首次回合窗口，不能被D的提前终止截短。',
        '总体均值':'对18个场景/种子的逐次指标等权平均；单位距离指标是各次比值的均值，不是总能量除以总距离。',
        '成功率':'保留原任务通过判据；方向阈值只作事后敏感性分析，不替代原成功率，不是预先注册标准。',
        '统计限制':'1个训练种子、3个评估种子、1套地形布局。重采样区间仅描述本批结果，不证明训练可重复性或真实电池节能。',
    }
    save(out/'gpt-B与C方向复评汇总.json', {'更新时间':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        '指标定义':definitions, 'B_C总体':summary, 'C相对B配对变化':changes, '逐场景':by_scene,
        '事后方向阈值敏感性':sensitivity, '两组共同窗口逐次指标':pairs,
        '代码SHA256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    write_csv(out/'gpt-五策略航向补充指标.csv',all_rows)
    write_csv(out/'gpt-B与C方向复评逐次指标.csv',[{k:v for k,v in row.items() if not isinstance(v,dict)} for row in pairs])
    write_csv(out/'gpt-B与C方向复评逐场景.csv',by_scene)
    write_csv(out/'gpt-B与C方向阈值敏感性.csv',sensitivity)
    plot(out,by_scene)
    lines=['# B与C方向保持及有效前进复评','',
        '已对原90场轨迹补充方向指标；B/C主比较为18组配对场景。无需重跑仿真，原始轨迹和既定任务通过判据保持原样。','',
        '## 指标与比较口径','']
    lines += [f'- **{name}**：{description}' for name,description in definitions.items()]
    lines += ['', 'B/C全部首次回合均完整，坡道5秒、其余10秒，每组总计165秒。原五策略共同窗口曾因D在崎岖种子47提前终止而截到9.2秒；本次B/C该场采用10秒，因此与旧均值的微小差别也包含窗口改变，不能全部归因于距离分母变化。','',
        '## B/C总体比较','', '| 指标 | B | C | C相对B变化 |','|---|---:|---:|---:|']
    for key,title in DISPLAY:
        item=changes[key]
        pct=item['C相对B变化百分比']
        lines.append(f'| {title} | {item["B"]:.4f} | {item["C"]:.4f} | {pct:+.2f}% |' if pct is not None else f'| {title} | {item["B"]:.4f} | {item["C"]:.4f} | 不可定义 |')
    lines += ['', '距离越大、距离利用比例越高通常越好；误差、偏移和单位距离代价越低越好。由于坡道时长短于其他场景，总体距离只用于同协议B/C比较，具体结论需结合分场景数据。','',
        '## 分场景方向与进展','', '| 场景 | 航向均方根 B/C（度） | 最大侧偏 B/C（米） | 有效前进 B/C（米） |','|---|---:|---:|---:|']
    for scene in SCENES:
        b,c=[next(r for r in by_scene if r['组别']==label and r['场景']==scene) for label in ('B','C')]
        ks=['航向偏差均方根度','最大绝对横向偏移米','沿初始方向有效前进距离米']
        lines.append(f'| {scene} | '+' | '.join(f'{b[k]:.3f} / {c[k]:.3f}' for k in ks)+' |')
    lines += ['', '## 按有效前进距离计算的能耗代理','',
        '| 场景 | 机械功 B/C（焦耳/有效米） | C相对B | 力矩平方积分 B/C（牛米平方秒/有效米） | C相对B |','|---|---:|---:|---:|---:|']
    for scene in SCENES:
        b,c=[next(r for r in by_scene if r['组别']==label and r['场景']==scene) for label in ('B','C')]
        ep,iq='每米有效前进机械功焦耳','每米有效前进力矩平方积分牛米平方秒每米'
        lines.append(f'| {scene} | {b[ep]:.3f} / {c[ep]:.3f} | {(c[ep]/b[ep]-1)*100:+.2f}% | {b[iq]:.3f} / {c[iq]:.3f} | {(c[iq]/b[iq]-1)*100:+.2f}% |')
    lines += ['', '## 事后方向阈值敏感性','',
        '以下为“原任务通过、全程无提前终止，并同时满足最大航向与侧偏限制”的场次数。阈值是在发现偏转后加入，用于透明展示敏感性，不将某一阈值包装为预先规定的成功率。','',
        '| 最大航向偏差上限（度） | 最大侧偏上限（米） | B | C |','|---|---:|---:|---:|']
    for row in sensitivity:
        lines.append(f'| {row["全程航向上限度"]:g} | {row["全程侧偏上限米"]:g} | {row["B满足次数"]}/18 | {row["C满足次数"]}/18 |')
    lines += ['', '## 数据驱动的结论','']
    for key in ('航向偏差均方根度','最大绝对横向偏移米','沿初始方向有效前进距离米','沿初始方向速度均方根误差米每秒','每米有效前进机械功焦耳','每米有效前进力矩平方积分牛米平方秒每米'):
        item=changes[key]; label=dict(DISPLAY)[key]
        lo,hi=item['配对种子差值95区间']
        lines.append(f'- {label}：C相对B变化 {item["C相对B变化百分比"]:+.2f}%；逐种子配对差值的描述性95%区间为 [{lo:.4f}, {hi:.4f}]。')
    ep,iq=changes['每米有效前进机械功焦耳'],changes['每米有效前进力矩平方积分牛米平方秒每米']
    if ep['C减B']>0 and iq['C减B']<0:
        lines += ['', '改用有效前进距离后，C仍以更高的机械功换取更低的力矩平方积分。B的机械能优势并未仅因改用新距离定义而消失，但方向保持代价必须同时报告；不能再把原任务通过次数相同解释为运动性能相同。']
    else:
        lines += ['', '应分别解释两类能耗代理的变化，不用未标定系数合成真实电耗结论。']
    lines += ['', '下一步建议：根据实际路线要求预先确定航向/侧偏容许范围，在更多独立训练种子上验证B/C，并补对应的纯机械功率低权重对照。当前方向指标补充与B/C离线复评已完成。', '',
        '![分场景比较](gpt-B与C方向保持及有效前进.png)', '']
    (out/'gpt-B与C方向保持及有效前进复评.md').write_text('\n'.join(lines))
