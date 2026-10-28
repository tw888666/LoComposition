"""自训30000轮模型按参考对话的取消质心偏移条件完整评估，保存真实地形。"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--gpu', default='2')
a = p.parse_args()
RUN = ROOT / 'gpt-复现记录/gpt-30000模型对照评估-20261021/完整评估'
TRAIN = ROOT / 'locomposition/logs/clean_rl/gpt-go2-default-seed46/2026-10-10-09-56-51'
RUN.mkdir(parents=True, exist_ok=True)
if list(RUN.glob('eval_*/metrics_summary.json')):
    raise SystemExit('已有完整结果，不覆盖。')
for name in ('params', 'model_30000.pt'):
    target = RUN / name
    if not target.exists():
        target.symlink_to(TRAIN / name, target_is_directory=(name == 'params'))
os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu, OMP_NUM_THREADS='8', OMNI_KIT_ACCEPT_EULA='Y',
                  OMNICLIENT_HUB_MODE='disabled', MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
scripts = ROOT / 'locomposition/scripts'
sys.path.insert(0, str(scripts))
sys.path.insert(0, str(ROOT / 'locomposition/exts/locomposition'))
spec = importlib.util.spec_from_file_location('gpt_full_eval', scripts / 'eval.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
original_hook = module.apply_common_eval_reward_scale_if_needed

def configure(env_cfg, task_name, enabled):
    value = original_hook(env_cfg, task_name, enabled)
    env_cfg.events.randomize_com.params['com_range'] = {axis: (0., 0.) for axis in 'xyz'}
    from isaaclab.terrains import TerrainImporter
    import numpy as np
    original_import = TerrainImporter.import_mesh
    def capture(self, name, mesh):
        np.savez_compressed(RUN / 'gpt-实际地形.npz', vertices=mesh.vertices, faces=mesh.faces)
        return original_import(self, name, mesh)
    TerrainImporter.import_mesh = capture
    return value

module.apply_common_eval_reward_scale_if_needed = configure
original_make = module.gym.make
def make(*args, **kwargs):
    from isaaclab.utils.io import dump_yaml
    dump_yaml(str(RUN / 'gpt-构造环境配置.yaml'), kwargs['cfg'].to_dict())
    env = original_make(*args, **kwargs)
    robot = env.unwrapped.scene['robot']
    (RUN / 'gpt-运行时质量与质心.json').write_text(json.dumps({
        '刚体名称': robot.data.body_names,
        '各刚体质量千克': robot.root_physx_view.get_masses()[0].cpu().tolist(),
        '各刚体局部质心位置与旋转': robot.root_physx_view.get_coms()[0].cpu().tolist(),
    }, ensure_ascii=False, indent=2))
    return env
module.gym.make = make
sys.argv = [str(scripts / 'eval.py'), '--run_dir', str(RUN), '--eval_checkpoint', '30000',
            '--seed', '46', '--task', 'LoComposition-Go2-Rough-Terrain-Joint-State-History-Play-v0',
            '--headless', '--no_video', '--skip_plots']
os.chdir(ROOT / 'locomposition')
module.main()
