"""从已通过核验的配对结果生成中文统计图和说明。"""
import importlib.util
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'gpt-复现记录/gpt-30000质心多种子配对评估-20261026'
SCENES = ('平地1.0', '平地1.6', '崎岖地形', '8.5厘米楼梯')
CONDITIONS = ('取消质心偏移', '保留质心偏移')
COLORS = ('#3678ae', '#d47a21')
spec = importlib.util.spec_from_file_location('stairs', ROOT / 'gpt-30000阶梯统计.py')
stairs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stairs)

def main():
    rows = json.loads((OUT / 'gpt-逐回合结果.json').read_text())
    summaries = json.loads((OUT / 'gpt-配对统计汇总.json').read_text())
    audits = json.loads((OUT / 'gpt-配对条件核验.json').read_text())
    assert len(rows) == 80 and audits['已核验配对'] == 40
    font = next((p for p in font_manager.findSystemFonts() if 'NotoSansCJK-Regular' in p), None)
    if font:
        font_manager.fontManager.addfont(font)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    labels = ['平地1.0', '平地1.6', '崎岖地形', '8.5厘米楼梯']
    x = np.arange(4)
    ax = axes[0, 0]
    for i, c in enumerate(CONDITIONS):
        counts = [s['条件统计'][c]['路线通过数'] for s in summaries[:4]]
        bars = ax.bar(x + (i - .5) * .34, counts, width=.32, color=COLORS[i], label=c)
        for bar, n in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, n + .15, f'{n}/10', ha='center', fontsize=10)
    ax.set_ylim(0, 11.5)
    ax.set_ylabel('路线通过次数')
    ax.set_title('任务完成：检查是否真正前进或越过楼梯')
    ax.legend(fontsize=10)
    for ax, key, title, unit in [
        (axes[0, 1], '前向速度均方根误差米每秒', '速度跟踪：越低越好', '米/秒'),
        (axes[1, 0], '横滚标准差度', '机身左右摇摆', '度'),
        (axes[1, 1], '俯仰标准差度', '机身前后俯仰波动', '度'),
    ]:
        for j, scene in enumerate(SCENES):
            for seed in range(40, 50):
                pair = [next(r for r in rows if r['场景'] == scene and r['种子'] == seed and r['条件'] == c)['共同窗口指标'][key] for c in CONDITIONS]
                ax.plot([j - .15, j + .15], pair, color='#a4aab0', alpha=.5, lw=.8)
                ax.scatter([j - .15, j + .15], pair, c=COLORS, s=15, alpha=.65)
            for i, c in enumerate(CONDITIONS):
                mean = summaries[j]['条件统计'][c][key]['均值']
                ax.scatter(j + (i - .5) * .30, mean, marker='D', s=60, color=COLORS[i], edgecolor='white', zorder=5)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.grid(axis='y', alpha=.2)
    for ax in axes.flat:
        ax.set_xticks(x, labels)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('第30000轮权重：保留与取消质心随机偏移的多种子配对评估', fontsize=17)
    fig.text(.06, .025, '10个种子 × 4场景 × 2条件；每回合10秒，固定地形布局，真实高程图开启。\n细线连接同种子配对值，菱形为均值；姿态统计剔除首秒，连续指标剔除重置帧并统一配对窗口。', fontsize=10)
    fig.tight_layout(rect=(0, .075, 1, .95))
    fig.savefig(OUT / 'gpt-质心偏移配对结果.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for ax, scene in zip(axes, SCENES[:3]):
        for i, c in enumerate(CONDITIONS):
            values = [next(r for r in rows if r['条件'] == c and r['种子'] == seed and r['场景'] == scene)['侧向偏移米'] for seed in range(40, 50)]
            ax.plot(range(40, 50), values, 'o-', color=COLORS[i], ms=4, lw=1.3, label=c)
        ax.axhline(1., ls='--', color='#b13434', lw=1.2, label='路线阈值1米')
        ax.set_title(scene)
        ax.set_xticks(range(40, 50, 2))
        ax.set_xlabel('随机种子')
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('首次重置前末帧侧向偏移（米）')
    axes[-1].legend(fontsize=9)
    fig.suptitle('路线偏移：区分身体稳定与持续沿指令方向前进', fontsize=15)
    fig.text(.06, .02, '两组均固定角速度指令为零，未启用外部航向纠偏；超过1米阈值不等同于跌倒。', fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, .92))
    fig.savefig(OUT / 'gpt-逐种子路线偏移.png', dpi=180)
    plt.close(fig)

    lines = ['第30000轮权重质心随机偏移配对评估', '',
             '测试已完成：10个种子（40—49），4个固定场景，两种条件，共80回合；每回合10秒。',
             '保留组：机身局部质心XYZ各轴在±3厘米范围启动时采样；取消组：同一采样调用，范围设零。',
             '地形布局固定，楼梯7级、每级8.5厘米；真实高程图始终开启。',
             '40对核验均通过：除质心位置外，环境配置、质量、材料、地形、初始状态、首步观测和共同窗口逐步随机数均一致。', '',
             '路线通过标准：平地和崎岖须无提前终止、前进≥目标距离70%、侧偏≤1米；楼梯须四足实际到达平台，满足位置和直立条件连续0.2秒。',
             '无提前终止单独统计，不把存活等同于任务完成。', '']
    intervals = []
    for item in summaries:
        lines.append(item['场景'])
        for c in CONDITIONS:
            s = item['条件统计'][c]
            lines.append(f"  {c}：路线通过{s['路线通过数']}/{s['回合数']}，无提前终止{s['无提前终止数']}/{s['回合数']}；全部所选场景通过的种子{s['十种子全部所选场景通过数']}/10。")
            if item['场景'] != '四场景等权平均':
                intervals.append({'条件': c, '场景': item['场景'], '路线通过数': s['路线通过数'], '测试数': 10,
                                  '路线通过率威尔逊95%区间': stairs.wilson(s['路线通过数'], 10),
                                  '无提前终止率威尔逊95%区间': stairs.wilson(s['无提前终止数'], 10)})
        for key in ('平均前向速度米每秒', '前向速度均方根误差米每秒', '横滚标准差度', '俯仰标准差度', '倾角95分位度', '每米绝对关节机械功焦耳'):
            a, b = [item['条件统计'][c][key] for c in CONDITIONS]
            delta = item['配对差值_保留减取消'][key]
            lo, hi = delta['配对自助法95%区间']
            lines.append(f"  {key}：取消{a['均值']:.4f}±{a['种子间标准差']:.4f}，保留{b['均值']:.4f}±{b['种子间标准差']:.4f}；配对差{delta['均值']:+.4f}，95%区间[{lo:+.4f}, {hi:+.4f}]。")
        lines.append('')
    failures = [r for r in rows if not r['路线通过'] or not r['无提前终止']]
    lines.append('未通过路线或提前终止的回合：')
    if not failures:
        lines.append('  无。')
    for r in failures:
        lines.append(f"  种子{r['种子']} {r['条件']} {r['场景']}：路线通过={r['路线通过']}，无提前终止={r['无提前终止']}，前进{r['前进米']:.4f}米，侧偏{r['侧向偏移米']:.4f}米，原因={r['提前终止原因']}。")
    lines += ['', '统计与解释边界：',
              '均值±标准差为种子间离散程度。配对区间以同种子保留减取消计算，按10个种子整组重采样10000次；不是把80回合当成80个独立种子。',
              '全部成功时成功率差的自助法区间会退化，不能解释为两组真实失败概率完全相同；每场景10/10成功的威尔逊95%区间约为72.2%—100%。',
              '本次只覆盖固定布局、10秒回合、训练范围内的质心偏移；不证明长期运行、未见地形、更高台阶、范围外配重或实机可靠性。',
              '机械功为关节绝对力矩乘角速度的积分，除以水平路径长度，不是电池能耗。',
              '本任务完成。下一步若需要扩大可靠性证据，应增加独立种子、连续运行时长和地形布局，并覆盖质心偏移边界。']
    (OUT / 'gpt-配对评估说明.txt').write_text('\n'.join(lines) + '\n')
    (OUT / 'gpt-成功率区间.json').write_text(json.dumps(intervals, ensure_ascii=False, indent=2) + '\n')
    print('统计图、中文说明和成功率区间已生成。')

if __name__ == '__main__':
    main()
