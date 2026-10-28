"""自训30000轮权重：复用21349七级台阶协议，原生加载训练权重与动作缩放。"""
import argparse,fcntl,importlib.util,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--height-cm',type=float,choices=(8.5,15,20),required=True)
parser.add_argument('--seed',type=int,required=True)
parser.add_argument('--gpu',default='2')
args=parser.parse_args()
OUT=ROOT/'gpt-复现记录/gpt-30000模型对照评估-20261021/阶梯评估'
RUN=OUT/f'{args.height_cm:g}厘米'/f'种子{args.seed}'
RUN.mkdir(parents=True,exist_ok=True)
TRAIN=ROOT/'locomposition/logs/clean_rl/gpt-go2-default-seed46/2026-10-10-09-56-51'
for name in ('params','model_30000.pt'):
    target=RUN/name
    if not target.exists():target.symlink_to(TRAIN/name,target_is_directory=(name=='params'))
run_lock=(RUN/'gpt-评估锁').open('a')
fcntl.flock(run_lock,fcntl.LOCK_EX)
if list(RUN.glob('eval_*/metrics_summary.json')):
    print('已有完整结果，不覆盖。',flush=True)
    raise SystemExit(0)
height=args.height_cm/100
spawn=[-8.,16.,-7*height+.4]
os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu,OMP_NUM_THREADS='8',OMNI_KIT_ACCEPT_EULA='Y',OMNICLIENT_HUB_MODE='disabled',MPLBACKEND='Agg',PYTHONUNBUFFERED='1')
scripts=ROOT/'locomposition/scripts'
sys.path.insert(0,str(scripts));sys.path.insert(0,str(ROOT/'locomposition/exts/locomposition'))
spec=importlib.util.spec_from_file_location('gpt_stair_eval',scripts/'eval.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)

def reset_fixed_spawn(env,env_ids):
    # 在环境重置阶段设置真实起点，确保首帧观测/历史在起点生成，避免先观测别处再瞬移。
    import torch
    robot=env.scene['robot']
    if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
    pose=torch.tensor(spawn+[0.,0.,0.,1.],device=env.device).repeat(len(env_ids),1)
    robot.write_root_pose_to_sim(pose,env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.zeros((len(env_ids),6),device=env.device),env_ids=env_ids)

original_config=module.apply_common_eval_reward_scale_if_needed

def configure(env_cfg,task_name,enabled):
    value=original_config(env_cfg,task_name,enabled)
    env_cfg.events.randomize_com.params['com_range']={axis:(0.,0.) for axis in 'xyz'}
    env_cfg.scene.terrain.terrain_generator.sub_terrains['pyramid_stairs_inv'].step_height_range=(height,height)
    env_cfg.events.reset_base.func=reset_fixed_spawn;env_cfg.events.reset_base.params={}
    env_cfg.episode_length_s=15.
    cmd=env_cfg.commands.base_velocity
    cmd.heading_command=False;cmd.rel_standing_envs=0.;cmd.rel_heading_envs=0.
    cmd.ranges.lin_vel_x=(1.,1.);cmd.ranges.lin_vel_y=(0.,0.);cmd.ranges.ang_vel_z=(0.,0.)
    from isaaclab.terrains import TerrainImporter
    import numpy as np
    original_import=TerrainImporter.import_mesh
    def capture(self,name,mesh):
        np.savez_compressed(RUN/'gpt-实际地形.npz',vertices=mesh.vertices,faces=mesh.faces)
        # 地形构造及平坦区域采样的随机消耗不应影响两种高度的配对随机化。
        module.set_global_seed(args.seed)
        return original_import(self,name,mesh)
    TerrainImporter.import_mesh=capture
    return value
module.apply_common_eval_reward_scale_if_needed=configure
original_seed=module.apply_seed_to_env_cfg

def seed_cfg(cfg,seed):
    original_seed(cfg,seed);cfg.scene.terrain.terrain_generator.seed=46
module.apply_seed_to_env_cfg=seed_cfg

def select_scenario(scenarios,selected,selected_set=None):
    import torch
    device=scenarios[0][1].device
    return [('fast_walk_stairs_up',torch.tensor([1.,0.,0.],device=device),(torch.tensor(spawn,device=device),torch.tensor([0.,0.,0.,1.],device=device)))]
module.select_fixed_command_scenarios=select_scenario
original_make=module.gym.make

def make(*positional,**keywords):
    from isaaclab.utils.io import dump_yaml
    dump_yaml(str(RUN/'gpt-构造环境配置.yaml'),keywords['cfg'].to_dict())
    env=original_make(*positional,**keywords);raw=env.unwrapped;robot=raw.scene['robot']
    (RUN/'gpt-运行时随机化.json').write_text(json.dumps({'种子':args.seed,'台阶厘米':args.height_cm,'刚体名称':robot.data.body_names,'各刚体质量千克':robot.root_physx_view.get_masses()[0].cpu().tolist(),'材料属性':robot.root_physx_view.get_material_properties()[0].cpu().tolist(),'局部质心':robot.root_physx_view.get_coms()[0].cpu().tolist()},ensure_ascii=False,indent=2))
    original_reset=raw._reset_idx;reset_records=[]
    def capture_reset(env_ids):
        if raw.episode_length_buf[0].item()>0:
            from isaaclab.utils.math import euler_xyz_from_quat
            d=robot.data;r,p,y=euler_xyz_from_quat(d.root_quat_w)
            record={'全局步':int(raw.common_step_counter),'回合步':int(raw.episode_length_buf[0]),'终止项':{n:bool(raw.termination_manager.get_term(n)[0]) for n in raw.termination_manager.active_terms},'机身位置':d.root_pos_w[0].cpu().tolist(),'机身角度偏航俯仰横滚':[y[0].item(),p[0].item(),r[0].item()],'关节角':d.joint_pos[0].cpu().tolist(),'足端世界坐标':[d.body_link_pos_w[0,d.body_names.index(leg+'_foot')].cpu().tolist() for leg in ('FL','FR','RL','RR')]}
            reset_records.append(record)
            (RUN/'gpt-重置前真实状态.json').write_text(json.dumps(reset_records,ensure_ascii=False,indent=2))
        return original_reset(env_ids)
    raw._reset_idx=capture_reset
    return env
module.gym.make=make
(RUN/'gpt-评估条件.json').write_text(json.dumps({'台阶高度米':height,'级数':7,'踏面深度米':.3,'总爬升米':7*height,'地形种子':46,'评估种子':args.seed,'起点':spawn,'起点机身离地米':.4,'机身朝向':'世界负X方向','指令速度米每秒':1.,'最长时间秒':15.,'质心随机偏移':'关闭；保留其余动力学和观测噪声','首帧初始化':'重置事件直接定位起点，固定速度指令在重置时即生效','随机数控制':'地形导入前重新设评估种子，隔离不同地形构造随机数消耗'},ensure_ascii=False,indent=2))
sys.argv=[str(scripts/'eval.py'),'--run_dir',str(RUN),'--seed',str(args.seed),'--task','LoComposition-Go2-Rough-Terrain-Joint-State-History-Play-v0','--headless','--no_video','--skip_plots','--eval_checkpoint','30000','--random_sim_step_length','0','--fixed_scenario','fast_walk_stairs_up','--fixed_command_sim_steps','750','--skip_cot_sweep']
os.chdir(ROOT/'locomposition')
module.main()
