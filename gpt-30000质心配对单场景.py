"""30000轮权重的质心随机偏移配对评估；两组保留真实高程图和同等随机数消耗。"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'gpt-复现记录/gpt-30000质心多种子配对评估-20261026'
SCENES = {
    '平地1.0': ('walk_x_flat_terrain_1.0mps', 1., [30.,30.,.4]),
    '平地1.6': ('walk_x_flat_terrain_1.6mps', 1.6, [30.,30.,.4]),
    '崎岖地形': ('fast_walk_x_uneven_terrain', 1., [0.,0.,.4]),
    '8.5厘米楼梯': ('fast_walk_stairs_up', 1., [-8.,16.,.4-7*.085]),
}
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--condition',choices=['取消质心偏移','保留质心偏移'],required=True)
p.add_argument('--scene',choices=list(SCENES),required=True)
p.add_argument('--gpu',default='3')
p.add_argument('--seed',type=int,default=46)
p.add_argument('--checkpoint',type=int,default=30000)
p.add_argument('--output-root',type=Path)
p.add_argument('--terrain-seed',type=int,default=46)
a=p.parse_args()
if a.output_root is not None:OUT=a.output_root.resolve()
tag,speed,spawn=SCENES[a.scene]
RUN=OUT/a.condition/a.scene
RUN.mkdir(parents=True,exist_ok=True)
lock=(RUN/'gpt-运行锁').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
if (RUN/'gpt-评估完成.json').exists():
    print('已有完成结果，不覆盖。',flush=True)
    raise SystemExit(0)
TRAIN=ROOT/'locomposition/logs/clean_rl/gpt-go2-default-seed46/2026-10-10-09-56-51'
model_name=f'model_{a.checkpoint}.pt'
assert (TRAIN/model_name).is_file(),f'权重不存在：{TRAIN/model_name}'
for name in ('params',model_name):
    if not (RUN/name).exists():(RUN/name).symlink_to(TRAIN/name,target_is_directory=name=='params')
os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='8',OMNI_KIT_ACCEPT_EULA='Y',
                  OMNICLIENT_HUB_MODE='disabled',MPLBACKEND='Agg',PYTHONUNBUFFERED='1')
scripts=ROOT/'locomposition/scripts'
sys.path.insert(0,str(scripts));sys.path.insert(0,str(ROOT/'locomposition/exts/locomposition'))
spec=importlib.util.spec_from_file_location('gpt_com_pair_eval',scripts/'eval.py')
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
raw_inputs=[];effective_inputs=[];applied_actions=[];rng_hashes=[];initial_states=[]

def rng_hash():
    import torch, numpy as np, random, pickle
    h=hashlib.sha256(torch.get_rng_state().numpy().tobytes())
    h.update(torch.cuda.get_rng_state().cpu().numpy().tobytes())
    h.update(pickle.dumps(np.random.get_state()))
    h.update(pickle.dumps(random.getstate()))
    return h.hexdigest()

def save_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def reset_fixed_spawn(env,env_ids):
    import torch
    if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
    robot=env.scene['robot']
    pose=torch.tensor(spawn+[0.,0.,0.,1.],device=env.device).repeat(len(env_ids),1)
    robot.write_root_pose_to_sim(pose,env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.zeros((len(env_ids),6),device=env.device),env_ids=env_ids)

original_hook=module.apply_common_eval_reward_scale_if_needed
def configure(env_cfg,task_name,enabled):
    value=original_hook(env_cfg,task_name,enabled)
    # 远程资源存在性检查偶发等待数分钟；先用HTTP检查同一URL，模型仍由原加载器读取。
    import importlib,subprocess
    files_module=importlib.import_module('isaaclab.sim.spawners.from_files.from_files')
    original_check=files_module.check_usd_path_with_timeout
    def check_asset(path,timeout=300,log_interval=30):
        if path.startswith('https://'):
            response=subprocess.run(['curl','-fsI','--max-time','15',path],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            if response.returncode==0:return True
        return original_check(path,timeout,log_interval)
    files_module.check_usd_path_with_timeout=check_asset
    assert env_cfg.events.randomize_com.mode=='startup'
    assert all(tuple(env_cfg.events.randomize_com.params['com_range'][axis])==(-.03,.03) for axis in 'xyz')
    if a.condition=='取消质心偏移':
        env_cfg.events.randomize_com.params['com_range']={axis:(0.,0.) for axis in 'xyz'}
    # 零范围仍执行原始随机采样，保留与非零范围一致的随机数消耗。
    env_cfg.events.reset_base.func=reset_fixed_spawn;env_cfg.events.reset_base.params={}
    if a.scene=='8.5厘米楼梯':
        env_cfg.scene.terrain.terrain_generator.sub_terrains['pyramid_stairs_inv'].step_height_range=(.085,.085)
    env_cfg.episode_length_s=10.
    cmd=env_cfg.commands.base_velocity
    cmd.heading_command=False;cmd.rel_standing_envs=0.;cmd.rel_heading_envs=0.
    cmd.ranges.lin_vel_x=(speed,speed);cmd.ranges.lin_vel_y=(0.,0.);cmd.ranges.ang_vel_z=(0.,0.)
    from isaaclab.terrains import TerrainImporter
    import numpy as np
    original_import=TerrainImporter.import_mesh
    def capture(self,name,mesh):
        np.savez_compressed(RUN/'gpt-实际地形.npz',vertices=mesh.vertices,faces=mesh.faces)
        module.set_global_seed(a.seed)
        return original_import(self,name,mesh)
    TerrainImporter.import_mesh=capture
    # 两组均使用完整真实观测，记录每步推理前随机数状态以核对配对。
    import locomposition.tasks.utils.cleanrl.ppo as ppo
    original_actor=ppo.ActorWithRMS
    class AuditedActor(original_actor):
        def forward(self,obs):
            assert obs.shape==(1,236)
            actual=obs.clone()
            rng_hashes.append(rng_hash())
            raw_inputs.append(obs[0].detach().cpu().numpy().copy())
            effective_inputs.append(actual[0].detach().cpu().numpy().copy())
            action=super().forward(actual)
            applied_actions.append(action[0].detach().cpu().numpy().copy())
            return action
    ppo.ActorWithRMS=AuditedActor
    return value
module.apply_common_eval_reward_scale_if_needed=configure
if a.terrain_seed is not None:
    original_seed_cfg=module.apply_seed_to_env_cfg
    def seed_cfg(cfg,seed):
        original_seed_cfg(cfg,seed)
        cfg.scene.terrain.terrain_generator.seed=a.terrain_seed
    module.apply_seed_to_env_cfg=seed_cfg

def select_scenario(scenarios,selected,selected_set=None):
    import torch
    device=scenarios[0][1].device
    return [(tag,torch.tensor([speed,0.,0.],device=device),
             (torch.tensor(spawn,device=device),torch.tensor([0.,0.,0.,1.],device=device)))]
module.select_fixed_command_scenarios=select_scenario
original_make=module.gym.make
def make(*args,**kwargs):
    from isaaclab.utils.io import dump_yaml
    import numpy as np
    dump_yaml(str(RUN/'gpt-构造环境配置.yaml'),kwargs['cfg'].to_dict())
    env=original_make(*args,**kwargs);raw=env.unwrapped;robot=raw.scene['robot']
    terms=raw.observation_manager.active_terms['policy']
    dims=raw.observation_manager.group_obs_term_dim['policy']
    assert terms[-1]=='height_map' and int(np.prod(dims[-1]))==143
    assert sum(int(np.prod(d)) for d in dims[:-1])==93
    save_json(RUN/'gpt-观测接口.json',{'观测顺序':terms,'各项维度':dims,'高程图切片':[93,236],
        '高程输入':'完整真实高程图；不替换任何策略观测'})
    save_json(RUN/'gpt-运行时随机化.json',{'刚体名称':robot.data.body_names,'构造后随机数摘要':rng_hash(),'各刚体质量千克':robot.root_physx_view.get_masses()[0].cpu().tolist(),
        '材料属性':robot.root_physx_view.get_material_properties()[0].cpu().tolist(),
        '局部质心':robot.root_physx_view.get_coms()[0].cpu().tolist()})
    original_public_reset=raw.reset
    def audited_public_reset(*args,**kwargs):
        output=original_public_reset(*args,**kwargs)
        initial_states.append({'机身位姿':robot.data.root_state_w[0,:13].cpu().tolist(),'关节角':robot.data.joint_pos[0].cpu().tolist(),'关节速度':robot.data.joint_vel[0].cpu().tolist(),'随机数摘要':rng_hash()})
        save_json(RUN/'gpt-初始状态.json',initial_states)
        return output
    raw.reset=audited_public_reset
    resets=[];original_reset=raw._reset_idx
    def capture_reset(env_ids):
        if raw.episode_length_buf[0].item()>0:
            from isaaclab.utils.math import euler_xyz_from_quat
            d=robot.data;r,p,y=euler_xyz_from_quat(d.root_quat_w)
            resets.append({'全局步':int(raw.common_step_counter),'回合步':int(raw.episode_length_buf[0]),
                '终止项':{n:bool(raw.termination_manager.get_term(n)[0]) for n in raw.termination_manager.active_terms},
                '机身位置':d.root_pos_w[0].cpu().tolist(),'机身角度偏航俯仰横滚':[y[0].item(),p[0].item(),r[0].item()],
                '关节角':d.joint_pos[0].cpu().tolist(),'前向速度':d.root_lin_vel_b[0,0].item(),
                '足端世界坐标':[d.body_link_pos_w[0,d.body_names.index(leg+'_foot')].cpu().tolist() for leg in ('FL','FR','RL','RR')]})
            save_json(RUN/'gpt-重置前真实状态.json',resets)
        return original_reset(env_ids)
    raw._reset_idx=capture_reset
    original_close=raw.close
    def close_with_audit(*args,**kwargs):
        # Kit关闭可能直接退出进程，必须在关闭环境之前写入审计结果。
        if raw_inputs:
            np.savez_compressed(RUN/'gpt-实际策略输入.npz',原始观测=np.array(raw_inputs),实际输入=np.array(effective_inputs),策略动作=np.array(applied_actions),随机数摘要=np.array(rng_hashes))
        if len(raw_inputs)==500 and list(RUN.glob('eval_*/metrics_summary.json')):
            save_json(RUN/'gpt-评估完成.json',{'状态':'完成','控制步数':500,'实际策略输入逐步保存':True})
        return original_close(*args,**kwargs)
    raw.close=close_with_audit
    return env
module.gym.make=make
save_json(RUN/'gpt-评估条件.json',{'条件':a.condition,'迭代':a.checkpoint,'场景':a.scene,'原场景名':tag,'指令速度米每秒':speed,
    '种子':a.seed,'地形种子':a.terrain_seed,'起点':spawn,'时长秒':10,'控制周期秒':.02,'台阶高度米':.085 if a.scene=='8.5厘米楼梯' else None,
    '高程图':'正常真实高程图','质心范围米':[-.03,.03] if a.condition=='保留质心偏移' else [0.,0.],
    '配对设计':'相同地形布局、相同随机种子、相同单场景流程；取消组保留随机采样调用，仅将采样范围设为零；逐步记录随机数状态。',
    '启动':'每个场景独立启动；重置阶段设置实际起点和固定指令，避免首帧旧观测。两组其余随机化相同。',
    '权重':str(TRAIN/model_name),'权重SHA256':hashlib.sha256((TRAIN/model_name).read_bytes()).hexdigest()} )
sys.argv=[str(scripts/'eval.py'),'--run_dir',str(RUN),'--eval_checkpoint',str(a.checkpoint),'--seed',str(a.seed),
    '--task','LoComposition-Go2-Rough-Terrain-Joint-State-History-Play-v0','--headless','--no_video','--skip_plots',
    '--random_sim_step_length','0','--fixed_scenario',tag,'--fixed_command_sim_steps','500','--skip_cot_sweep']
os.chdir(ROOT/'locomposition')
try:
    module.main()
finally:
    import numpy as np
    if raw_inputs:
        np.savez_compressed(RUN/'gpt-实际策略输入.npz',原始观测=np.array(raw_inputs),实际输入=np.array(effective_inputs),策略动作=np.array(applied_actions),随机数摘要=np.array(rng_hashes))
assert len(raw_inputs)==500 and list(RUN.glob('eval_*/metrics_summary.json'))
save_json(RUN/'gpt-评估完成.json',{'状态':'完成','控制步数':500,'实际策略输入逐步保存':True})
