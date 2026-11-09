"""G 组（B 的权重 0.012 加 20% 力矩平方混合，mix_alpha=0.2）训练种子 46/47/48 三个等预算微调任务及配对评估。

结构沿用 claude-BC组合F多训练种子流水线.py：同一个 30000 次预训练起点、5000 次微调、7500 环境、学习率 3e-5，
训练与评估都通过冻结实现运行。冻结实现相对 C/E 复验只多了分支 F/G 和评估标签 F/G。
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import importlib.util
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'claude-BC组合G多训练种子-20261109'
FROZEN = OUT/'冻结实现'
LAUNCHER = ROOT/'claude-BC组合G冻结代码启动.py'
spec = importlib.util.spec_from_file_location('gpt_ce_old_helpers', ROOT/'gpt-纯机械功率消融流水线.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
save, read, sha, now = old.save, old.read, old.sha, old.now
JOBS = (('G', 46, 2), ('G', 47, 3), ('G', 48, 4))
WEIGHT, ALPHA = 0.012, 0.2  # = 0.008*1.5，混合 20% 力矩平方
STARTUP_LOCK = threading.Lock()


def verify_code():
    for record in read(OUT/'claude-实验代码清单.json')['文件']:
        assert sha(ROOT/record['相对路径']) == record['SHA256'], '冻结代码或固定输入变化：'+record['相对路径']


def freeze_code():
    """记录冻结实现、启动器、评估入口和固定输入的 SHA256；正式训练前生成，之后每个任务开始前核对。"""
    files = sorted(p for p in FROZEN.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files += [LAUNCHER, OUT/'claude-冻结评估入口.py', Path(__file__).resolve(), old.BASE, old.REFERENCE]
    save(OUT/'claude-实验代码清单.json', {'冻结时间': now(), '文件': [
        {'相对路径': str(p.resolve().relative_to(ROOT)), 'SHA256': sha(p)} for p in files]})


def audit_training(run, seed, smoke, import_path):
    import torch
    iterations, envs = (3, 64) if smoke else (5000, 7500)
    m, p = read(run/'gpt-四组微调配置.json'), read(run/'gpt-微调来源.json')
    assert m['实验组'] == 'G' and m['随机种子'] == seed
    assert m['追加迭代'] == iterations and m['环境数量'] == envs and m['每轮控制步'] == 24
    assert p['环境随机种子'] == seed and p['并行环境数'] == envs and p['微调总迭代'] == iterations
    assert p['来源文件SHA256'] == sha(old.BASE) == read(old.REFERENCE)['checkpoint_sha256']
    assert p['逐键完全一致'] and len(p['已核验状态键']) == 23
    assert not p['优化器已恢复'] and p['优化器状态条目数'] == 0
    actual = p['实际运行参数']
    assert abs(actual['energy_weight']-WEIGHT) < 1e-14 and actual['mix_alpha'] == ALPHA
    assert actual['scaling_factor'] == 1 and actual['torque_scale'] == read(old.REFERENCE)['s']
    assert actual['learning_rate'] == 3e-5 and actual['constraint_progress'] == 1
    # 与 C 组训练种子 46 的历史运行对照：只允许种子、实验名和能耗项不同
    previous = Path(read(old.OLD/'gpt-C-状态.json')['模型目录'])
    ref = read(previous/'gpt-微调来源.json')
    assert p['成熟约束最大终止概率'] == ref['成熟约束最大终止概率']
    env = old.config(run/'params/env.yaml')
    agent = old.config(run/'params/agent.yaml')
    assert env['seed'] == env['sim']['random_seed'] == env['scene']['terrain']['terrain_generator']['seed'] == seed
    assert agent['seed'] == seed
    env_diff = old.differences(old.config(previous/'params/env.yaml'), env)
    agent_diff = old.differences(old.config(previous/'params/agent.yaml'), agent)
    seed_keys = {'.seed', '.sim.random_seed', '.scene.terrain.terrain_generator.seed'} if seed != 46 else set()
    energy_keys = {'.rewards.minimize_power.weight', '.curriculum.power.params.start_weight',
                   '.curriculum.power.params.end_weight', '.rewards.minimize_power.params.mix_alpha'}
    allowed_env = seed_keys | energy_keys | ({'.scene.num_envs'} if smoke else set())
    allowed_agent = ({'.seed'} if seed != 46 else set()) | {'.experiment_name'} | ({'.num_iterations'} if smoke else set())
    assert set(row[0] for row in env_diff) <= allowed_env, env_diff
    assert set(row[0] for row in agent_diff) <= allowed_agent, agent_diff
    model = torch.load(run/'model_1.pt', map_location='cpu', weights_only=True)
    base = torch.load(old.BASE, map_location='cpu', weights_only=True)
    assert model.keys() == base.keys() and all(torch.isfinite(v).all() for v in model.values())
    changed = sum(not torch.equal(model[k], base[k]) for k in model)
    assert changed
    imports = read(import_path)
    assert imports['状态'] == '通过' and imports['实际项目模块全部来自冻结目录']
    return {'状态': '通过', '训练种子': seed, '组别': 'G', '正式训练': not smoke, '时间': now(),
            '初始权重逐项一致': True, '首轮已更新且有限': True, '更新键数': changed,
            '实际运行参数': actual, '与C组种子46环境差异': env_diff, '算法差异': agent_diff,
            '实际项目模块全部来自冻结目录': True, '模型目录': str(run)}


def environment(seed, gpu, smoke):
    env = old.environment(gpu, smoke)
    env['ENV_NAME'] = f'claude-energy-G-seed{seed}-'+('smoke-' if smoke else '')+'20261109'
    env['PYTHONPATH'] = str(FROZEN/'exts/locomposition')+os.pathsep+env.get('PYTHONPATH', '')
    return env


def job(label, seed, gpu, smoke):
    tag = f'{label}-训练种子{seed}'
    folder = OUT/('冒烟' if smoke else '正式任务')/tag
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder/'claude-运行锁').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = folder/'claude-状态.json'
    state = {'状态': '执行中', '阶段': '训练中', '组别': label, '训练种子': seed, '物理GPU': gpu,
             '开始时间': now(), '正式实验': not smoke, '已完成迭代': 0, '评估完成场数': 0}
    process = None
    startup_locked = False
    try:
        verify_code()
        iterations, envs = (3, 64) if smoke else (5000, 7500)
        import_path = folder/'claude-训练导入来源.json'
        command = [str(old.PYTHON), str(LAUNCHER), 'train', '--audit', str(import_path), '--',
                   '--task=LoComposition-Go2-Rough-Terrain-Joint-State-History-v0', f'--seed={seed}', '--headless',
                   f'--num_envs={envs}', f'--num_iterations={iterations}', '--logger=tensorboard', '--device=cuda:0',
                   '--finetune', f'--checkpoint={old.BASE}', f'--energy_branch={label}', f'--energy_reference={old.REFERENCE}',
                   'env.scene.ray_caster.debug_vis=false', 'env.scene.ray_caster_height_constraints.debug_vis=false',
                   'env.commands.base_velocity.debug_vis=false']
        log_path = folder/'claude-训练.log'
        state.update(训练指令=command, 训练日志=str(log_path), 追加迭代=iterations)
        state['阶段'] = '等待环境初始化'; save(path, state)
        # 共享地形缓存先写网格再写原点；首轮完成前避免另一个任务读取半写入缓存。
        STARTUP_LOCK.acquire(); startup_locked = True
        state['阶段'] = '环境初始化与首轮训练'
        with log_path.open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT/'locomposition', env=environment(seed, gpu, smoke),
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state['训练进程PID'] = process.pid
            save(path, state)
            run, audited = None, False
            while True:
                with log_path.open(errors='replace') as stream:
                    beginning = stream.read(1024*1024)
                    stream.seek(max(0, log_path.stat().st_size-65536))
                    ending = stream.read()
                matches = re.findall(r'\[INFO\] Reserved run directory: (.+)', beginning)
                if matches:
                    run = Path(matches[-1]); state['模型目录'] = str(run)
                if run and not audited and (run/'model_1.pt').is_file() and import_path.is_file():
                    save(folder/'claude-启动核验.json', audit_training(run, seed, smoke, import_path))
                    audited = True; state['启动核验'] = '通过'
                    state['阶段'] = '训练中'
                    STARTUP_LOCK.release(); startup_locked = False
                progress = re.findall(r'\[训练\] 迭代=(\d+)/\d+ 耗时=([\d.]+)s', ending)
                if progress:
                    state['已完成迭代'] = int(progress[-1][0])
                    state['近期每轮秒'] = sum(float(r[1]) for r in progress[-10:])/len(progress[-10:])
                state['更新时间'] = now(); save(path, state)
                code = process.poll()
                if code is not None:
                    assert code == 0 and audited and run and (run/f'model_{iterations}.pt').is_file(), f'{tag}训练未正常完成，退出码{code}'
                    break
                time.sleep(10)
        state.update(阶段='评估中', 已完成迭代=iterations, 训练完成时间=now(), 最终权重SHA256=sha(run/f'model_{iterations}.pt'))
        save(path, state)
        output = (OUT/'冒烟评估' if smoke else OUT)/f'训练种子{seed}'
        pairs = [(46, '平地1.0')] if smoke else [(s, c) for s in old.SEEDS for c in old.SCENES]
        for evaluation_seed, scene in pairs:
            verify_code()
            imported = folder/f'claude-评估导入来源-{evaluation_seed}-{scene}.json'
            command = [str(old.PYTHON), str(LAUNCHER), 'eval', '--audit', str(imported), '--',
                       '--run-dir', str(run), '--label', label, '--checkpoint', str(iterations), '--gpu', str(gpu),
                       '--seed', str(evaluation_seed), '--scene', scene, '--output-root', str(output)]
            if smoke: command += ['--steps', '12']
            state['当前评估'] = {'评估种子': evaluation_seed, '场景': scene}; save(path, state)
            with (folder/f'claude-评估-{evaluation_seed}-{scene}.log').open('a') as log:
                subprocess.run(command, cwd=ROOT, env=environment(seed, gpu, smoke), stdout=log, stderr=subprocess.STDOUT, check=True)
            assert read(imported)['状态'] == '通过'
            state['评估完成场数'] += 1; save(path, state)
        state.update(状态='冒烟完成' if smoke else '训练评估完成', 阶段='完成', 完成时间=now()); save(path, state)
        return state
    except BaseException as error:
        state.update(状态='失败，需检查', 错误=str(error), 失败时间=now()); save(path, state)
        raise
    finally:
        if startup_locked:
            STARTUP_LOCK.release()
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--freeze', action='store_true', help='只生成冻结代码清单后退出')
    args = parser.parse_args()
    if args.freeze:
        freeze_code(); print('已生成冻结代码清单'); return
    lock = (OUT/'claude-总运行锁').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = OUT/('claude-冒烟总状态.json' if args.smoke else 'claude-总状态.json')
    state = {'状态': '执行中', '开始时间': now(), '正式实验': not args.smoke, '主进程PID': os.getpid(), '完成任务': [], '错误': []}
    if not args.smoke:
        assert os.environ.get('TMUX'), '正式训练必须在tmux中运行'
        assert read(OUT/'claude-冒烟总状态.json')['状态'] == '全部完成'
    save(path, state)
    with ThreadPoolExecutor(max_workers=len(JOBS)) as pool:
        pending = {pool.submit(job, *item, args.smoke): item[:2] for item in JOBS}
        for future in as_completed(pending):
            tag = pending[future]
            try:
                future.result(); state['完成任务'].append(list(tag))
            except Exception as error:
                state['错误'].append({'任务': tag, '错误': str(error)})
            save(path, state)
    if state['错误']:
        state.update(状态='存在失败，需检查', 结束时间=now()); save(path, state)
        raise RuntimeError('部分任务失败')
    state.update(状态='全部完成', 完成时间=now()); save(path, state)


if __name__ == '__main__':
    main()

