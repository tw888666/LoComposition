"""五策略统一评估：真实高程、训练电机、冻结归一化、确定性动作和重置前逐步数据。"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / 'gpt-四组微调实验-20261026'
SCENES = {
    '平地0.4': ('walk_x_flat_terrain_0.4mps', .4, [30., 30., .4], 500),
    '平地0.8': ('cot_sweep_walk_x_flat_terrain_0.8', .8, [30., 30., .4], 500),
    '平地1.0': ('walk_x_flat_terrain_1.0mps', 1., [30., 30., .4], 500),
    '崎岖1.0': ('fast_walk_x_uneven_terrain', 1., [0., 0., .4], 500),
    '楼梯8.5厘米': ('fast_walk_stairs_up', 1., [-8., 16., .4 - 7 * .085], 500),
    # 使用种子46现有地形中的约6.8度坡道，5秒覆盖上坡及平台，避免驶入相邻随机地形。
    '坡道0.8': ('walk_existing_slope_0.8mps', .8, [-4.7, -16., .45], 250),
}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--label', choices=('A0', 'A', 'B', 'C', 'D'), required=True)
    p.add_argument('--checkpoint', type=int, required=True)
    p.add_argument('--gpu', default='4')
    p.add_argument('--seed', type=int, default=46)
    p.add_argument('--scene', choices=SCENES)
    p.add_argument('--output-root', type=Path, default=DEFAULT_OUT)
    p.add_argument('--steps', type=int, help='仅用于冒烟验证；改变正式时长时结果自动标记为冒烟。')
    a = p.parse_args()
    source = a.run_dir.resolve()
    checkpoint = source / f'model_{a.checkpoint}.pt'
    assert checkpoint.is_file(), checkpoint
    assert (source / 'params/env.yaml').is_file(), source / 'params/env.yaml'
    if a.scene is None:
        for scene in SCENES:
            cmd = [sys.executable, str(Path(__file__).resolve()), '--run-dir', str(source),
                   '--label', a.label, '--checkpoint', str(a.checkpoint), '--gpu', a.gpu,
                   '--seed', str(a.seed), '--scene', scene, '--output-root', str(a.output_root)]
            if a.steps is not None:
                cmd += ['--steps', str(a.steps)]
            subprocess.run(cmd, cwd=ROOT, check=True)
        return
    tag, speed, spawn, planned_steps = SCENES[a.scene]
    steps = planned_steps if a.steps is None else a.steps
    assert steps > 0
    run = a.output_root.resolve() / '评估' / a.label / f'检查点{a.checkpoint}' / f'种子{a.seed}' / a.scene
    if steps != planned_steps:
        run = run / f'冒烟{steps}步'
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / 'gpt-运行锁').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    if (run / 'gpt-评估完成.json').exists():
        old = json.loads((run / 'gpt-评估条件.json').read_text())
        assert old['权重SHA256'] == sha(checkpoint), '已有结果所用权重发生变化，拒绝覆盖。'
        print('已有完成结果：', run, flush=True)
        return
    for name in ('params', checkpoint.name):
        target = run / name
        if target.is_symlink() or target.exists():
            assert target.resolve() == (source / name).resolve(), target
        else:
            target.symlink_to(source / name, target_is_directory=name == 'params')

    os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu, OMP_NUM_THREADS='8', OMNI_KIT_ACCEPT_EULA='Y',
                      OMNICLIENT_HUB_MODE='disabled', MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
    scripts = ROOT / 'locomposition/scripts'
    sys.path.insert(0, str(scripts))
    sys.path.insert(0, str(ROOT / 'locomposition/exts/locomposition'))
    spec = importlib.util.spec_from_file_location('gpt_finetune_eval', scripts / 'eval.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    import numpy as np
    raw_inputs, actions, resets, states, constraints, actors = [], [], [], {}, {}, []

    def reset_fixed_spawn(env, env_ids):
        import torch
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        robot = env.scene['robot']
        pose = torch.tensor(spawn + [0., 0., 0., 1.], device=env.device).repeat(len(env_ids), 1)
        robot.write_root_pose_to_sim(pose, env_ids=env_ids)
        robot.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids)

    original_hook = module.apply_common_eval_reward_scale_if_needed

    def configure(env_cfg, task_name, enabled):
        value = original_hook(env_cfg, task_name, enabled)
        # 保留已验证的远程资源存在性检查；资源仍由原始加载器读取。
        import importlib
        files_module = importlib.import_module('isaaclab.sim.spawners.from_files.from_files')
        original_check = files_module.check_usd_path_with_timeout

        def check_asset(path, timeout=300, log_interval=30):
            if path.startswith('https://'):
                reply = subprocess.run(['curl', '-fsI', '--max-time', '15', path],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if reply.returncode == 0:
                    return True
            return original_check(path, timeout, log_interval)
        files_module.check_usd_path_with_timeout = check_asset
        env_cfg.events.randomize_com.params['com_range'] = {axis: (0., 0.) for axis in 'xyz'}
        env_cfg.events.reset_base.func = reset_fixed_spawn
        env_cfg.events.reset_base.params = {}
        if a.scene == '楼梯8.5厘米':
            env_cfg.scene.terrain.terrain_generator.sub_terrains['pyramid_stairs_inv'].step_height_range = (.085, .085)
        env_cfg.episode_length_s = steps * .02
        cmd = env_cfg.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_standing_envs = 0.
        cmd.rel_heading_envs = 0.
        cmd.ranges.lin_vel_x = (speed, speed)
        cmd.ranges.lin_vel_y = (0., 0.)
        cmd.ranges.ang_vel_z = (0., 0.)
        from isaaclab.terrains import TerrainImporter
        original_import = TerrainImporter.import_mesh

        def capture_mesh(self, name, mesh):
            np.savez_compressed(run / 'gpt-实际地形.npz', vertices=mesh.vertices, faces=mesh.faces)
            if a.scene == '坡道0.8':
                v = np.asarray(mesh.vertices)
                measured = {}
                for x, expected in ((-4.7, .05), (-6., .205), (-7., .235)):
                    candidates = v[np.linalg.norm(v[:, :2] - [x, -16.], axis=1) < .001, 2]
                    assert len(candidates) and np.isclose(candidates.max(), expected, atol=.006), (x, candidates)
                    measured[str(x)] = float(candidates.max())
                save(run / 'gpt-坡道几何核验.json', {'世界Y米': -16., '各世界X处地面高度米': measured,
                     '来源': '种子46原有正金字塔坡道，平台0.235米，局部上坡约6.8度。',
                     '时长理由': '5秒覆盖上坡和平台；避免持续10秒进入相邻随机地形。'})
            module.set_global_seed(a.seed)
            return original_import(self, name, mesh)
        TerrainImporter.import_mesh = capture_mesh
        import locomposition.tasks.utils.cleanrl.ppo as ppo
        original_actor = ppo.ActorWithRMS

        class AuditedActor(original_actor):
            def __init__(self, trained_agent):
                super().__init__(trained_agent)
                self.eval()
                self._frozen_mean = self.running_mean.detach().cpu().numpy().copy()
                self._frozen_var = self.running_var.detach().cpu().numpy().copy()
                actors.append(self)

            def forward(self, obs):
                assert obs.shape == (1, 236)
                raw_inputs.append(obs[0].detach().cpu().numpy().copy())
                action = super().forward(obs)
                actions.append(action[0].detach().cpu().numpy().copy())
                return action
        ppo.ActorWithRMS = AuditedActor
        return value
    module.apply_common_eval_reward_scale_if_needed = configure
    original_seed_cfg = module.apply_seed_to_env_cfg

    def seed_cfg(cfg, seed):
        original_seed_cfg(cfg, seed)
        cfg.scene.terrain.terrain_generator.seed = 46
    module.apply_seed_to_env_cfg = seed_cfg

    def select_scenario(scenarios, selected, selected_set=None):
        import torch
        device = scenarios[0][1].device
        return [(tag, torch.tensor([speed, 0., 0.], device=device),
                 (torch.tensor(spawn, device=device), torch.tensor([0., 0., 0., 1.], device=device)))]
    module.select_fixed_command_scenarios = select_scenario
    original_make = module.gym.make

    def flush_audit():
        if not raw_inputs:
            return
        for actor in actors:
            np.testing.assert_array_equal(actor.running_mean.detach().cpu().numpy(), actor._frozen_mean)
            np.testing.assert_array_equal(actor.running_var.detach().cpu().numpy(), actor._frozen_var)
        np.savez_compressed(run / 'gpt-实际策略输入.npz', 原始观测=np.array(raw_inputs),
                            实际输入=np.array(raw_inputs), 策略动作=np.array(actions),
                            归一化均值=actors[0]._frozen_mean, 归一化方差=actors[0]._frozen_var)
        data = {key: np.array(values) for key, values in states.items()}
        data.update({'约束__' + key: np.array(values) for key, values in constraints.items()})
        np.savez_compressed(run / 'gpt-控制步真实数据.npz', **data)
        save(run / 'gpt-重置前真实状态.json', resets)
        if len(raw_inputs) == steps and len(states.get('关节力矩', [])) == steps and list(run.glob('eval_*/metrics_summary.json')):
            save(run / 'gpt-评估完成.json', {'状态': '完成', '控制步数': steps, '归一化冻结核验': True,
                 '动作': '策略均值，未采样动作噪声', '重置前逐步数据': True, '正式评估': steps == planned_steps})

    def make(*args, **kwargs):
        from isaaclab.utils.io import dump_yaml
        from isaaclab.utils.math import euler_xyz_from_quat
        import torch
        dump_yaml(str(run / 'gpt-构造环境配置.yaml'), kwargs['cfg'].to_dict())
        env = original_make(*args, **kwargs)
        raw = env.unwrapped
        robot = raw.scene['robot']
        terms = raw.observation_manager.active_terms['policy']
        dims = raw.observation_manager.group_obs_term_dim['policy']
        assert terms[-1] == 'height_map' and int(np.prod(dims[-1])) == 143
        assert sum(int(np.prod(d)) for d in dims[:-1]) == 93
        save(run / 'gpt-观测接口.json', {'观测顺序': terms, '各项维度': dims, '高程图切片': [93, 236], '高程': '真实高程不遮蔽'})
        save(run / 'gpt-运行时随机化.json', {'各刚体质量千克': robot.root_physx_view.get_masses()[0].cpu().tolist(),
             '材料属性': robot.root_physx_view.get_material_properties()[0].cpu().tolist(),
             '局部质心': robot.root_physx_view.get_coms()[0].cpu().tolist()})
        save(run / 'gpt-运行时电机配置.json', raw.cfg.scene.robot.actuators['base_legs'].to_dict())
        feet = [robot.data.body_names.index(leg + '_foot') for leg in ('FL', 'FR', 'RL', 'RR')]
        sensor = raw.scene['contact_forces']
        foot_sensor_ids, _ = sensor.find_bodies([leg + '_foot' for leg in ('FL', 'FR', 'RL', 'RR')], preserve_order=True)
        original_compute = raw.constraint_manager.compute

        def compute_with_trace(*args, **kwargs):
            value = original_compute(*args, **kwargs)
            d = robot.data
            roll, pitch, yaw = euler_xyz_from_quat(d.root_quat_w)
            values = {'关节力矩': d.applied_torque[0], '关节角速度': d.joint_vel[0],
                      '关节位置': d.joint_pos[0], '关节加速度': d.joint_acc[0],
                      '机身位置': d.root_pos_w[0], '机身姿态偏航俯仰横滚': torch.stack((yaw[0], pitch[0], roll[0])),
                      '机身线速度机体系': d.root_lin_vel_b[0], '机身角速度机体系': d.root_ang_vel_b[0],
                      '足端世界坐标': d.body_link_pos_w[0, feet],
                      '足端接触力': torch.linalg.vector_norm(sensor.data.net_forces_w_history[0][:, foot_sensor_ids, :], dim=-1).amax(dim=0),
                      '指令速度': raw.command_manager.get_command('base_velocity')[0],
                      '动作': raw.action_manager.action[0], '终止': raw.reset_terminated[0],
                      '超时': raw.reset_time_outs[0]}
            for key, tensor in values.items():
                states.setdefault(key, []).append(tensor.detach().cpu().numpy().copy())
            for key, tensor in raw.constraint_manager.cat.raw_constraints.items():
                constraints.setdefault(key, []).append(tensor[0].detach().cpu().numpy().copy())
            return value
        raw.constraint_manager.compute = compute_with_trace
        original_reset = raw._reset_idx

        def capture_reset(env_ids):
            if raw.episode_length_buf[0].item() > 0:
                d = robot.data
                roll, pitch, yaw = euler_xyz_from_quat(d.root_quat_w)
                resets.append({'全局步': int(raw.common_step_counter), '回合步': int(raw.episode_length_buf[0]),
                     '终止项': {n: bool(raw.termination_manager.get_term(n)[0]) for n in raw.termination_manager.active_terms},
                     '机身位置': d.root_pos_w[0].cpu().tolist(),
                     '机身角度偏航俯仰横滚': [yaw[0].item(), pitch[0].item(), roll[0].item()],
                     '关节角': d.joint_pos[0].cpu().tolist(), '足端世界坐标': d.body_link_pos_w[0, feet].cpu().tolist()})
            return original_reset(env_ids)
        raw._reset_idx = capture_reset
        original_close = raw.close

        def close_with_audit(*args, **kwargs):
            flush_audit()
            return original_close(*args, **kwargs)
        raw.close = close_with_audit
        return env
    module.gym.make = make
    save(run / 'gpt-评估条件.json', {'组别': a.label, '检查点': a.checkpoint, '场景': a.scene,
         '原场景名': tag, '指令速度米每秒': speed, '种子': a.seed, '地形种子': 46,
         '起点': spawn, '初始航向弧度': float(np.pi), '初始根姿态四元数wxyz': [0., 0., 0., 1.],
         '时长秒': steps * .02, '控制周期秒': .02, '正式评估': steps == planned_steps,
         '台阶高度米': .085 if '楼梯' in a.scene else None,
         '电机': 'UNITREE_GO2_CFG_TRAIN（训练电机配置）', '高程图': '真实143维高程',
         '归一化': '冻结各权重自身的观测归一化统计', '动作': '确定性策略均值',
         '随机化': '取消质心随机偏移，保留其余评估随机化；地形生成后重设评估种子。',
         '权重': str(checkpoint), '权重SHA256': sha(checkpoint),
         '物理数据口径': '每控制步物理推进后、环境重置前记录；功率为控制频率采样的机械功率代理。'})
    sys.argv = [str(scripts / 'eval.py'), '--run_dir', str(run), '--eval_checkpoint', str(a.checkpoint),
                '--seed', str(a.seed), '--task', 'LoComposition-Go2-Rough-Terrain-Joint-State-History-Play-v0',
                '--headless', '--no_video', '--skip_plots', '--random_sim_step_length', '0',
                '--fixed_scenario', tag, '--fixed_command_sim_steps', str(steps), '--skip_cot_sweep',
                '--use_training_go2_config']
    os.chdir(ROOT / 'locomposition')
    try:
        module.main()
    finally:
        flush_audit()
    assert (run / 'gpt-评估完成.json').exists(), '评估未完整保存。'


if __name__ == '__main__':
    main()
