"""不启动仿真的原始TorchScript（导出策略）独立接口验证。"""
from pathlib import Path
import hashlib
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "locomposition/scripts"))
from torchscript_policy import load_exported_policy

MODEL = ROOT / "locomposition/sim2real/traced_checkpoints/2025-06-28-17-13-04_21349_traced_deterministic.pt"


class AuthorPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.metadata = load_exported_policy(MODEL)

    def test_independent_forward_and_frozen_normalization(self):
        state = {key: value.clone() for key, value in self.model.state_dict().items()}
        generator = torch.Generator().manual_seed(46)
        x = state["running_mean"] + torch.randn((32, 236), generator=generator) * torch.sqrt(state["running_var"] + 1e-8)
        expected = (x - state["running_mean"]) / torch.sqrt(state["running_var"] + 1e-8)
        for i in (0, 2, 4, 6):
            expected = torch.nn.functional.linear(expected, state[f"actor.{i}.weight"], state[f"actor.{i}.bias"])
            if i != 6:
                expected = torch.nn.functional.elu(expected)
        with torch.inference_mode():
            actual = self.model(x)
            repeated = self.model(x)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual, repeated, atol=0, rtol=0)
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, state[key], atol=0, rtol=0)
        self.assertEqual(actual.shape, (32, 12))

    def test_wrong_observation_dimension_is_rejected(self):
        with self.assertRaises((RuntimeError, torch.jit.Error)):
            self.model(torch.zeros(1, 188))

    def test_source_model_unchanged(self):
        self.assertEqual(hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                         "ac81a9195f4c55461c8bcb4b6ccfc61f7d6cdc509176f64421110875d7efcfac")


if __name__ == "__main__":
    unittest.main(verbosity=2)
