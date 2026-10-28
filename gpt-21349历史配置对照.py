"""基于2025年6月公开源码的作者模型配置诊断；不冒充原训练参数。"""
import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "gpt-复现记录/gpt-21349历史配置核查-20261019"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("condition", choices=["仅取消质心偏移", "历史公开评估配置", "当前配置核对"])
parser.add_argument("--gpu", default="2")
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()
run = OUT / (("冒烟-" if args.smoke else "") + args.condition)
if list(run.glob("eval_*/metrics_summary.json")):
    raise SystemExit("已有结果，请使用新的条件目录，禁止覆盖。")
run.mkdir(parents=True, exist_ok=True)
os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS="8", OMNI_KIT_ACCEPT_EULA="Y",
                  OMNICLIENT_HUB_MODE="disabled", MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
scripts = ROOT / "locomposition/scripts"
sys.path.insert(0, str(scripts))
sys.path.insert(0, str(ROOT / "locomposition/exts/locomposition"))
spec = importlib.util.spec_from_file_location("gpt_historical_eval", scripts / "eval.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

original_config_hook = module.apply_common_eval_reward_scale_if_needed
def configure(env_cfg, task_name, enabled):
    value = original_config_hook(env_cfg, task_name, enabled)
    changes = []
    if args.condition == "仅取消质心偏移":
        # 保留事件及采样次数，只将分布压为零，减少随机数流变化。
        env_cfg.events.randomize_com.params["com_range"] = {axis: (0., 0.) for axis in "xyz"}
        changes.append("仅将机身质心偏移范围由各轴±0.03米改为零；保留事件采样及其他配置。")
    elif args.condition == "历史公开评估配置":
        from isaaclab.envs import mdp
        from isaaclab.managers import EventTermCfg, SceneEntityCfg
        env_cfg.events.randomize_com = None
        env_cfg.events.randomize_mass = None
        env_cfg.events.episodic_height_offset = None
        env_cfg.events.reset_base.params["pose_range"]["yaw"] = (-1.57, 1.57)
        env_cfg.events.reset_robot_joints = EventTermCfg(
            func=mdp.reset_joints_by_scale, mode="reset",
            params={"position_range": (.95, 1.05), "velocity_range": (-.05, .05)})
        env_cfg.observations.policy.enable_corruption = False
        env_cfg.apply_elevation_map_pose_noise = False
        env_cfg.apply_elevation_map_point_noise = False
        env_cfg.scene.ray_caster.ray_cast_drift_range = {axis: (0., 0.) for axis in "xyz"}
        motor = env_cfg.scene.robot.actuators["base_legs"]
        motor.effort_limit = 23.5
        motor.saturation_effort = 23.5
        motor.velocity_limit = 30.
        motor.friction = 0.
        # 这些旧顶层属性沿用历史源码；不擅自将其映射到不同的physx属性。
        env_cfg.sim.max_position_iteration_count = 4
        env_cfg.sim.max_velocity_iteration_count = 1
        # 精确使用导出的历史高度函数；无需导入旧包或执行旧模块的全局随机种子代码。
        source = (OUT / "gpt-训练时间附近-环境.txt").read_text()
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "height_map_grid")
        namespace = {"torch": module.torch, "SceneEntityCfg": SceneEntityCfg, "__name__": __name__}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<历史高度函数>", "exec"), namespace)
        historical_height = namespace["height_map_grid"]
        globals()["height_map_grid"] = historical_height
        policy = env_cfg.observations.policy
        policy.height_map.func = historical_height
        policy.height_map.params = {"asset_cfg": SceneEntityCfg("ray_caster")}
        policy.base_ang_vel.func = mdp.base_ang_vel
        policy.base_ang_vel.params = {}
        policy.projected_gravity.func = mdp.projected_gravity
        policy.projected_gravity.params = {}
        policy.actions.func = mdp.last_action
        policy.actions.params = {}
        changes += ["取消2026年新增的质心、质量和回合高度偏移事件。", "恢复历史重置角度范围和默认站姿附近±5%关节随机初始化。",
                    "关闭评估观测噪声及新增射线漂移；使用历史高度函数和直接角速度、重力、动作观测。",
                    "恢复历史评估电机23.5/23.5/30/0，保持25/0.5控制增益。",
                    "保留当前仿真器、机器人资产、地形实现和统一14500步评估协议，不声称重建2025年的整个软件栈。"]
    else:
        changes.append("配置完全不变，仅记录当前默认环境的实际质量与质心，作为诊断参考。")
    (run / "gpt-条件说明.json").write_text(json.dumps({"条件": args.condition, "依据提交": "2d1daae",
        "原训练配置": "未获得，仅由训练时间附近公开代码提出配置假设", "修改": changes,
        "种子": 46, "物理显卡": args.gpu}, ensure_ascii=False, indent=2))
    return value
module.apply_common_eval_reward_scale_if_needed = configure

original_make = module.gym.make
def make(*positional, **keywords):
    from isaaclab.utils.io import dump_yaml
    dump_yaml(str(run / "gpt-构造环境配置.yaml"), keywords["cfg"].to_dict())
    env = original_make(*positional, **keywords)
    robot = env.unwrapped.scene["robot"]
    coms = robot.root_physx_view.get_coms()[0].cpu().tolist()
    masses = robot.root_physx_view.get_masses()[0].cpu().tolist()
    (run / "gpt-运行时质量与质心.json").write_text(json.dumps({"刚体名称": robot.data.body_names,
        "各刚体质量千克": masses, "各刚体局部质心位置与旋转": coms},ensure_ascii=False,indent=2))
    return env
module.gym.make = make
sys.argv = [str(scripts / "eval.py"), "--run_dir", str(run), "--seed", "46", "--task",
            "LoComposition-Go2-Rough-Terrain-Joint-State-History-Play-v0", "--headless", "--no_video", "--skip_plots",
            "--torchscript_path", str(ROOT / "locomposition/sim2real/traced_checkpoints/2025-06-28-17-13-04_21349_traced_deterministic.pt"),
            "--torchscript_action_scale", "0.8"]
if args.smoke:
    sys.argv += ["--random_sim_step_length", "0", "--fixed_scenario", "walk_x_flat_terrain_1.0mps", "--fixed_command_sim_steps", "100", "--skip_cot_sweep"]
os.chdir(ROOT / "locomposition")
module.main()
