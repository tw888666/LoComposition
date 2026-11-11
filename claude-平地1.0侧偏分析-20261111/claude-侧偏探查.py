"""探查：平地 1.0 侧偏的来源。只读已有评估轨迹，不训练。"""
import itertools, json
from pathlib import Path
import numpy as np
ROOT = Path('/home/xy.chen/tw/LoComposition')
OLD = ROOT/'gpt-四组微调实验-20261026/评估'
E46 = ROOT/'gpt-纯机械功率消融-20261028/评估'
CE = ROOT/'gpt-CE多训练种子复验-20261102'
F = ROOT/'claude-BC组合F多训练种子-20261105'
G = ROOT/'claude-BC组合G多训练种子-20261109'
def root(ts, g):
    if g == 'F': return F/f'训练种子{ts}/评估'
    if g == 'G': return G/f'训练种子{ts}/评估'
    if ts == 46: return E46 if g == 'E' else OLD
    return CE/f'训练种子{ts}/评估'
MODELS = [(46, g) for g in 'ABCEFG'] + [(ts, g) for ts in (47, 48) for g in 'CEFG']
TH = 5.0
def analyse(run):
    d = np.load(run/'gpt-控制步真实数据.npz'); cfg = json.loads((run/'gpt-评估条件.json').read_text())
    r = np.flatnonzero(d['终止'] | d['超时']); n = int(r[0]+1) if len(r) else len(d['终止'])
    dt = .02; h = np.pi
    yaw = np.unwrap(d['机身姿态偏航俯仰横滚'][:n, 0]); err = yaw - h; err = (err + np.pi) % (2*np.pi) - np.pi
    err = np.unwrap(err)
    pos = d['机身位置'][:n, :2] - np.array(cfg['起点'][:2])
    lat = pos @ np.array([-np.sin(h), np.cos(h)])
    vx, vy = d['机身线速度机体系'][:n, 0], d['机身线速度机体系'][:n, 1]
    # 世界侧向速度 = vx·sin(err) + vy·cos(err)：前者来自航向偏差，后者来自机体系侧滑
    from_heading = np.cumsum(vx*np.sin(err))*dt; from_slip = np.cumsum(vy*np.cos(err))*dt
    w = np.degrees(d['机身角速度机体系'][:n, 2])
    c = d['足端接触力'][:n] > TH
    m = np.ones(n, bool); m[:50] = False
    d1 = m & c[:, 0] & c[:, 3] & ~c[:, 1] & ~c[:, 2]; d2 = m & c[:, 1] & c[:, 2] & ~c[:, 0] & ~c[:, 3]
    P = np.abs(d['关节力矩'][:n]*d['关节角速度'][:n])
    L = P[:, [0, 2, 4, 6, 8, 10]].sum(1).mean(); R = P[:, [1, 3, 5, 7, 9, 11]].sum(1).mean()
    k = lambda t: float(np.degrees(err[min(int(t/dt), n-1)]))
    return dict(n=n, 末帧航向=float(np.degrees(err[-1])), 航向2s=k(2), 航向5s=k(5), 末帧侧偏=float(lat[-1]),
                航向致侧偏=float(from_heading[-1]), 侧滑致侧偏=float(from_slip[-1]), 平均侧滑=float(vy[m].mean()),
                平均偏航率=float(w[m].mean()), FLRR偏航=float(w[d1].mean()), FRRL偏航=float(w[d2].mean()),
                左右功率差百分比=float((L-R)/(L+R)*100), 平均前向速度=float(vx[m].mean()))
rows = []
for (ts, g), s, sc in itertools.product(MODELS, (46, 47, 48), ('平地0.4', '平地0.8', '平地1.0', '崎岖1.0')):
    rows.append(dict(训练种子=ts, 组别=g, 评估种子=s, 场景=sc, **analyse(root(ts, g)/g/'检查点5000'/f'种子{s}'/sc)))
Path(ROOT/'claude-平地1.0侧偏分析-20261111/claude-逐次侧偏分解.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1))
for sc in ('平地1.0',):
    print('==', sc, '  列：末帧航向° | 末帧侧偏m = 航向致 + 侧滑致 | 平均偏航率°/s | FL+RR / FR+RL °/s | 左右功率差%')
    for r in rows:
        if r['场景'] != sc: continue
        print(f"{r['训练种子']} {r['组别']} 评{r['评估种子']}: {r['末帧航向']:+6.1f} | {r['末帧侧偏']:+5.2f} = {r['航向致侧偏']:+5.2f} + {r['侧滑致侧偏']:+5.2f} | "
              f"{r['平均偏航率']:+5.2f} | {r['FLRR偏航']:+6.2f} / {r['FRRL偏航']:+6.2f} | {r['左右功率差百分比']:+5.1f}")
