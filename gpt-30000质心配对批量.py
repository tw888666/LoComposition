"""十种子、两种质心条件、四场景评估；独立进程、可断点继续。"""
import argparse
import concurrent.futures
import hashlib
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'gpt-复现记录/gpt-30000质心多种子配对评估-20261026'
MODEL = ROOT / 'locomposition/logs/clean_rl/gpt-go2-default-seed46/2026-10-10-09-56-51/model_30000.pt'
SCENES = ('平地1.0', '平地1.6', '崎岖地形', '8.5厘米楼梯')
CONDITIONS = ('取消质心偏移', '保留质心偏移')

def write(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)

def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        '权重路径': str(MODEL), '权重SHA256': hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        '种子': list(range(40, 50)), '条件': CONDITIONS, '场景': SCENES,
        '地形种子': 46, '每回合秒数': 10, '每回合控制步数': 500, '控制周期秒': .02,
        '测试总数': 80, '独立种子对数': 10,
        '唯一条件差异': '机身局部质心三轴各±0.03米，或者三轴均为零；两组均调用同一采样函数，随机数消耗一致。',
        '流程': '每个种子、条件、场景均独立启动进程；无随机指令前缀。真实高程图始终开启。两组均在重置阶段定位和设置指令。',
        '地形范围': '固定同一布局，楼梯固定为7级、每级8.5厘米；不代表不同地形布局和更高楼梯。',
        '存活定义': '整个10秒窗口没有提前自动重置，也没有记录到非超时终止项；不等同于路线通过。',
        '平地与崎岖通过定义': '无提前终止，世界负X净前进至少为指令速度×10秒的70%，末帧侧向偏移不超过1米。',
        '楼梯通过定义': '首次重置前，四足X均≤-11.05米、足高均≥-0.03米、足Y均在13.5至18.5米、机身Z≥0.2米、倾角≤30度，连续满足10步。另独立报告通过后是否提前终止。',
        '连续指标窗口': '每一对测试使用首次自动重置前的共同窗口；剔除重置帧。速度含起步，姿态波动另外排除首1秒。',
        '连续指标': ['实际前向速度', '前向速度均方根误差', '横滚与俯仰标准差', '倾角95分位数', '每米绝对关节机械功'],
        '统计': '先计算同种子同场景的保留减取消差值；按种子配对重采样10000次计算95%区间。四场景同种子整组重采样；80回合不是80个独立种子。',
        '核验': ['同权重', '同地形网格', '配置除质心范围外一致', '质量与材料一致', '仅base局部质心位置变化', '初始状态一致', '首步观测一致', '共同窗口逐步随机数摘要一致'],
        '限制': '每场景每条件仅10次、每次10秒，不能证明零失败概率、长时间可靠性或实机表现。',
    }
    path = OUT / 'gpt-预定评估协议.json'
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(protocol))
    else:
        write(path, protocol)
    jobs = []
    for seed in range(40, 50):
        for scene in SCENES:
            for condition in CONDITIONS:
                jobs.append({'种子': seed, '场景': scene, '条件': condition,
                             '目录': str(OUT / f'种子{seed}' / condition / scene)})
    write(OUT / 'gpt-运行清单.json', jobs)
    return jobs

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--gpus', default='3,4')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    a = parser.parse_args()
    jobs = prepare()
    if a.prepare_only:
        return
    if a.smoke:
        jobs = [j for j in jobs if j['种子'] == 46 and j['场景'] == '平地1.0']
    gpus = a.gpus.split(',')
    pending = queue.Queue()
    done, failed = [], []
    lock = threading.Lock()
    for j in jobs:
        if (Path(j['目录']) / 'gpt-评估完成.json').exists():
            done.append(j)
        else:
            pending.put(j)
    def worker(index):
        gpu = gpus[index % len(gpus)]
        while True:
            try:
                j = pending.get_nowait()
            except queue.Empty:
                return
            label = f"种子{j['种子']}-{j['条件']}-{j['场景']}"
            started = time.monotonic()
            attempts = []
            ok = False
            for attempt in range(1, 3):
                log = OUT / f'gpt-{label}-运行{attempt}.log'
                cmd = [sys.executable, str(ROOT / 'gpt-30000质心配对单场景.py'),
                       '--condition', j['条件'], '--scene', j['场景'], '--seed', str(j['种子']),
                       '--terrain-seed', '46', '--gpu', gpu,
                       '--output-root', str(OUT / f"种子{j['种子']}")]
                print('开始', label, '显卡', gpu, '尝试', attempt, flush=True)
                with log.open('w') as f:
                    try:
                        code = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=1200).returncode
                    except subprocess.TimeoutExpired:
                        code = '超时'
                ok = code == 0 and (Path(j['目录']) / 'gpt-评估完成.json').exists()
                attempts.append({'次数': attempt, '退出码': code, '日志': str(log)})
                if ok:
                    break
            state = {**j, '成功完成': ok, '设备': gpu, '耗时秒': time.monotonic() - started, '进程记录': attempts}
            write(OUT / f'gpt-{label}-进程状态.json', state)
            with lock:
                (done if ok else failed).append(j)
                write(OUT / ('gpt-冒烟进度.json' if a.smoke else 'gpt-批量进度.json'),
                      {'计划': len(jobs), '完成': len(done), '运行失败': len(failed), '最近完成': state})
                print('进度', len(done), '/', len(jobs), '运行失败', len(failed), label, round(state['耗时秒'], 1), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for future in concurrent.futures.as_completed([pool.submit(worker, i) for i in range(a.workers)]):
            future.result()
    write(OUT / ('gpt-冒烟执行结果.json' if a.smoke else 'gpt-批量执行结果.json'),
          {'完成': len(done), '总计': len(jobs), '运行失败': failed})
    if failed:
        raise RuntimeError('有仿真进程失败，不能当作机器人测试失败。可断点继续。')

if __name__ == '__main__':
    main()
