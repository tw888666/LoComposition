"""同一模型的小规模诊断：默认评估、训练电机参数、横向指令范围内。

所有结果写入独立目录；不修改作者评估器或训练模型。
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("condition", choices=["默认评估", "训练电机", "范围内横向指令"])
parser.add_argument("--repeat", action="store_true", help="独立复测目录，不覆盖首轮结果")
args = parser.parse_args()
source = ROOT / "locomposition/logs/clean_rl/gpt-go2-default-seed46/2026-10-10-09-56-51"
run = ROOT / "gpt-复现记录/gpt-姿态速度诊断-20261019" / (("复测-" if args.repeat else "") + args.condition)
run.mkdir(parents=True, exist_ok=True)
for name in ["model_30000.pt", "params"]:
    link = run / name
    if not link.exists(): link.symlink_to(source / name, target_is_directory=name == "params")
if list(run.glob("eval_*/metrics_summary.json")):
    raise SystemExit("本条件已有结果，拒绝覆盖。")
os.environ.update(CUDA_VISIBLE_DEVICES="2", OMP_NUM_THREADS="8", OMNI_KIT_ACCEPT_EULA="Y",
                  OMNICLIENT_HUB_MODE="disabled", MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
scripts = ROOT / "locomposition/scripts"
sys.path.insert(0, str(scripts))
# 顶层仓库也名为 locomposition；显式优先扩展包，避免命名空间遮蔽。
sys.path.insert(0, str(ROOT / "locomposition/exts/locomposition"))
spec = importlib.util.spec_from_file_location("gpt_diagnostic_eval", scripts / "eval.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

def choose(scenarios, *_args):
    names = ["walk_x_flat_terrain_1.0mps", "walk_x_flat_terrain_1.6mps", "fast_walk_diagonal_uneven_terrain"]
    selected = [next(row for row in scenarios if row[0] == name) for name in names]
    if args.condition == "范围内横向指令":
        name, cmd, pose = selected[-1]
        cmd = cmd.clone(); cmd[1] = .7
        selected[-1] = (name, cmd, pose)
    (run / "gpt-诊断条件.json").write_text(json.dumps({
        "条件": args.condition, "模型": str(source / "model_30000.pt"), "种子": 46,
        "每场景步数": 500, "随机前缀步数": 0,
        "场景指令": {name: cmd.cpu().tolist() for name, cmd, _ in selected},
        "说明": "电机条件仅替换机器人训练配置；横向条件仅改变第三场景vy，其他场景及随机种子相同。此为因果诊断，不代替论文正式评估。"
    }, ensure_ascii=False, indent=2))
    return selected

module.select_fixed_command_scenarios = choose
samples = {key: [] for key in ["computed_effort", "applied_effort", "gravity", "motor_joint_velocity"]}
reset_records = []
original_make = module.gym.make
def monitored_make(*positional, **keywords):
    env = original_make(*positional, **keywords)
    robot = env.unwrapped.scene["robot"]
    actuator = robot.actuators["base_legs"]
    (run / "gpt-运行时电机配置.json").write_text(json.dumps({
        "电机配置": actuator.cfg.to_dict(),
        "关节顺序": robot.data.joint_names,
        "默认关节角": robot.data.default_joint_pos[0].cpu().tolist(),
        "物理关节速度限制": robot.data.joint_vel_limits[0].cpu().tolist(),
    }, ensure_ascii=False, indent=2, default=str))
    original_step = env.step
    def monitored_step(action):
        value = original_step(action)
        samples["computed_effort"].append(actuator.computed_effort[0].detach().cpu().numpy().copy())
        samples["applied_effort"].append(actuator.applied_effort[0].detach().cpu().numpy().copy())
        samples["motor_joint_velocity"].append(actuator._joint_vel[0].detach().cpu().numpy().copy())
        samples["gravity"].append(robot.data.projected_gravity_b[0].cpu().numpy().copy())
        if env.unwrapped.episode_length_buf[0].item() == 0:
            reset_records.append({"步骤": len(samples['gravity'])-1,
                                  "重置信息": {k: str(v) for k,v in value[-1].get('log', {}).items() if 'Termination' in k}})
        # 仿真器关闭可能直接结束解释器，所以在控制循环内保存，不能依赖 main 返回。
        if len(samples['gravity']) % 500 == 0:
            module.np.savez_compressed(run / "gpt-电机与姿态诊断.npz", **samples)
            (run / 'gpt-提前重置诊断.json').write_text(json.dumps(reset_records,ensure_ascii=False,indent=2))
        return value
    env.step = monitored_step
    return env
module.gym.make = monitored_make
sys.argv = [str(scripts / "eval.py"), "--run_dir", str(run), "--seed", "46",
            "--task", "LoComposition-Go2-Rough-Terrain-Joint-State-History-Play-v0",
            "--headless", "--no_video", "--skip_plots", "--skip_cot_sweep",
            "--random_sim_step_length", "0"]
if args.condition == "训练电机": sys.argv.append("--use_training_go2_config")
os.chdir(ROOT / "locomposition")
module.main()
module.np.savez_compressed(run / "gpt-电机与姿态诊断.npz", **samples)
