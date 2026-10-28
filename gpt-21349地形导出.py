"""按原评估种子与配置导出真实地形，用于已记录轨迹的视频回放。"""
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "gpt-复现记录/gpt-21349崎岖与楼梯视频-20261019"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.update(CUDA_VISIBLE_DEVICES="2", OMP_NUM_THREADS="8", OMNI_KIT_ACCEPT_EULA="Y",
                  OMNICLIENT_HUB_MODE="disabled", MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
scripts = ROOT / "locomposition/scripts"
sys.path.insert(0, str(scripts))
sys.path.insert(0, str(ROOT / "locomposition/exts/locomposition"))
spec = importlib.util.spec_from_file_location("gpt_terrain_eval", scripts / "eval.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
original_hook = module.apply_common_eval_reward_scale_if_needed

def configure(env_cfg, task_name, enabled):
    value = original_hook(env_cfg, task_name, enabled)
    env_cfg.events.randomize_com.params["com_range"] = {axis: (0., 0.) for axis in "xyz"}
    return value

module.apply_common_eval_reward_scale_if_needed = configure
def export_without_robot(*args, **kwargs):
    # 地形生成不依赖机器人加载；避开远端机器人资产服务的等待。
    from isaaclab.terrains import TerrainGenerator
    from isaaclab.utils.io import dump_yaml
    import numpy as np
    cfg = kwargs["cfg"]
    dump_yaml(str(OUT / "gpt-地形重建环境配置.yaml"), cfg.to_dict())
    terrain = TerrainGenerator(cfg.scene.terrain.terrain_generator, device="cpu")
    mesh = terrain.terrain_mesh
    np.savez_compressed(OUT / "gpt-原仿真地形.npz", vertices=mesh.vertices, faces=mesh.faces,
                        terrain_origins=terrain.terrain_origins)
    print("真实地形已导出", mesh.vertices.shape, mesh.faces.shape, flush=True)
    raise SystemExit(0)

module.gym.make = export_without_robot
sys.argv = [str(scripts / "eval.py"), "--run_dir", str(OUT / "地形核对运行"), "--seed", "46", "--task",
            "LoComposition-Go2-Rough-Terrain-Joint-State-History-Play-v0", "--headless", "--no_video", "--skip_plots",
            "--torchscript_path", str(ROOT / "locomposition/sim2real/traced_checkpoints/2025-06-28-17-13-04_21349_traced_deterministic.pt"),
            "--torchscript_action_scale", "0.8", "--random_sim_step_length", "0", "--fixed_scenario",
            "walk_x_flat_terrain_1.0mps", "--fixed_command_sim_steps", "10", "--skip_cot_sweep"]
os.chdir(ROOT / "locomposition")
module.main()
