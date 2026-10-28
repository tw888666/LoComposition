"""在不启动仿真器的情况下验证检查点续训契约。"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
import torch

source = Path(__file__).resolve().parent / 'locomposition/exts/locomposition/locomposition/tasks/utils/cleanrl/checkpoints.py'
spec = importlib.util.spec_from_file_location('gpt_checkpoints', source)
checkpoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoint)


def make_env():
    terrain = NS(terrain_levels=torch.tensor([1, 0]), terrain_types=torch.tensor([0, 1]),
                 terrain_origins=torch.arange(12).reshape(2, 2, 3).float(), env_origins=torch.zeros(2, 3))
    return NS(common_step_counter=240, _sim_step_counter=960, cfg=NS(decimation=4), device='cpu',
              curriculum_manager=NS(constraints_curriculum=0.75), scene=NS(terrain=terrain),
              constraint_manager=NS(cat=NS(running_maxes={'joint_torque': torch.tensor([[3.]])})))


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)
        self.agent = torch.nn.Linear(2, 1)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=0.0003)
        self.agent(torch.ones(4, 2)).sum().backward()
        self.optimizer.step()

    def test_old_model_requires_explicit_partial_resume(self):
        path = self.path / 'model_1249.pt'
        checkpoint.atomic_save(self.agent.state_dict(), path)
        with self.assertRaisesRegex(ValueError, 'allow_partial_resume'):
            checkpoint.load_training_checkpoint(path, self.agent, self.optimizer,
                device='cpu', total_iterations=30000)
        target = torch.nn.Linear(2, 1)
        opt = torch.optim.Adam(target.parameters())
        state = checkpoint.load_training_checkpoint(path, target, opt, device='cpu',
            total_iterations=30000, allow_partial=True, constraint_progress=1.0)
        self.assertEqual(state['iteration'], 1249)
        self.assertFalse(opt.state)
        self.assertTrue(torch.equal(target.weight, self.agent.weight))
        env = make_env()
        checkpoint.restore_environment_progress(env, state, 24)
        self.assertEqual(env.common_step_counter, 1249 * 24)
        self.assertEqual(env._sim_step_counter, 1249 * 24 * 4)
        self.assertEqual(env.curriculum_manager.constraints_curriculum, 1.0)

    def test_training_state_round_trip(self):
        path = self.path / 'training_state_10.pt'
        checkpoint.save_training_checkpoint(path, self.agent, self.optimizer, make_env(), 10, 30000)
        target = torch.nn.Linear(2, 1)
        opt = torch.optim.Adam(target.parameters())
        state = checkpoint.load_training_checkpoint(path, target, opt, device='cpu', total_iterations=30000)
        self.assertFalse(state['partial_resume'])
        self.assertTrue(torch.equal(target.weight, self.agent.weight))
        for saved, loaded in zip(self.optimizer.state.values(), opt.state.values()):
            self.assertTrue(torch.equal(saved['exp_avg'], loaded['exp_avg']))
            self.assertTrue(torch.equal(saved['exp_avg_sq'], loaded['exp_avg_sq']))
            self.assertTrue(torch.equal(saved['step'], loaded['step']))
        env = make_env()
        env.scene.terrain.terrain_levels.zero_()
        checkpoint.restore_environment_progress(env, state, 24)
        self.assertEqual(env.scene.terrain.terrain_levels.tolist(), [1, 0])
        self.assertEqual(env.scene.terrain.env_origins.tolist(), [[6., 7., 8.], [3., 4., 5.]])
        self.assertEqual(env.curriculum_manager.constraints_curriculum, 0.75)
        self.assertEqual(env.common_step_counter, 240)
        self.assertFalse(path.with_name(path.name + '.tmp').exists())
        with self.assertRaisesRegex(ValueError, '总迭代预算'):
            checkpoint.load_training_checkpoint(path, target, opt, device='cpu', total_iterations=15000)

    def test_invalid_checkpoint_rejected(self):
        path = self.path / 'model_30000.pt'
        checkpoint.atomic_save(self.agent.state_dict(), path)
        with self.assertRaises(ValueError):
            checkpoint.load_training_checkpoint(path, self.agent, self.optimizer, device='cpu',
                total_iterations=30000, allow_partial=True)
        path = self.path / 'model_10.pt'
        with torch.no_grad():
            self.agent.weight.fill_(float('nan'))
        checkpoint.atomic_save(self.agent.state_dict(), path)
        with self.assertRaisesRegex(ValueError, '非有限'):
            checkpoint.load_training_checkpoint(path, self.agent, self.optimizer, device='cpu',
                total_iterations=30000, allow_partial=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
