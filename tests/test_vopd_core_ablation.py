"""Core ablations of Scheme A: no clip, constant beta, and no 1/(1-beta) normalization."""

import math
from pathlib import Path
import unittest

from omegaconf import OmegaConf
import torch

from modules.vOPD.src.token_gate import (dense_token_gate_loss, token_terms, validate_ablation,
                                         validate_gate)
from scripts._runtime import load_config, validate_assets


ROOT = Path(__file__).resolve().parents[1]
CLIP = .16944029927253723
INF = math.inf
CONFIGS = {
    "01_jsd_q05_noclip": ("token_jsd", INF, None, True),
    "02_mix_q05_noclip": ("token_mix", INF, None, True),
    "03_jsd_beta05": ("token_jsd", INF, .5, True),
    "04_fkl": ("token_jsd", INF, 1., True),
    "05_jsd_q05_unnorm": ("token_jsd", CLIP, None, False),
}


def log(rows):
    return torch.tensor(rows, dtype=torch.float64).log()


def kl(a, b):
    return (a.exp() * (a - b)).sum(-1)


class CoreAblationTests(unittest.TestCase):
    def test_infinite_clip_equals_unclipped_objective(self):
        for kind in ("token_jsd", "token_mix"):
            s = log([[.1, .9], [.7, .3], [.01, .99]]).requires_grad_()
            t = log([[.8, .2], [.8, .2], [.8, .2]])
            _, clipped = token_terms(s, t, .5, kind, False, [1], CLIP)
            self.assertGreater(clipped[:, 8].sum(), 0)
            reference, _ = token_terms(s, t, .5, kind, False, [1], None)
            actual, stats = token_terms(s, t, .5, kind, False, [1], INF)
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            self.assertEqual(stats[:, 8].sum(), 0)
            torch.testing.assert_close(torch.autograd.grad(actual.sum(), s, retain_graph=True)[0],
                                       torch.autograd.grad(reference.sum(), s)[0], rtol=0, atol=0)

    def test_gate_accepts_infinite_but_not_invalid_clip(self):
        validate_gate("token_jsd", .5, False, [1], 1, INF)
        for clip in (float("nan"), 0., -1., -INF):
            with self.assertRaises(ValueError):
                validate_gate("token_jsd", .5, False, [1], 1, clip)

    def test_constant_beta_is_normalized_jsd_at_every_position(self):
        p = log([[.1, .8, .1], [.4, .3, .3]]).requires_grad_()
        q = log([[.8, .15, .05], [.4, .35, .25]])  # The second position fails tau_q=0.5.
        _, gated = token_terms(p, q, .5, "token_jsd", False, [2], INF)
        self.assertEqual(gated[:, 1].tolist(), [1., 0.])
        loss, stats = token_terms(p, q, .5, "token_jsd", False, [2], INF, fixed_beta=.5)
        m = torch.logaddexp(p, q) - math.log(2)
        torch.testing.assert_close(loss, 2 * kl(q, m) + 2 * kl(p, m))
        torch.testing.assert_close(stats[:, 0], torch.full((2,), .5, dtype=torch.float64))
        torch.testing.assert_close(stats[:, 1], torch.ones(2, dtype=torch.float64))

    def test_constant_beta_one_is_exact_fkl(self):
        p = log([[.1, .8, .1], [.4, .3, .3]]).requires_grad_()
        q = log([[.8, .15, .05], [.4, .35, .25]])
        loss, stats = token_terms(p, q, .5, "token_jsd", False, [2], INF, fixed_beta=1.)
        torch.testing.assert_close(loss, kl(q, p))
        torch.testing.assert_close(torch.autograd.grad(loss.sum(), p)[0], (p.exp() - q.exp()).detach())
        self.assertEqual(stats[:, 7].abs().sum(), 0)

    def test_unnormalized_weights_clipped_jsd_by_one_minus_beta(self):
        # Clipped disagreement, unclipped disagreement, agreement, and an ungated position.
        s = log([[.01, .98, .01], [.3, .6, .1], [.75, .2, .05], [.1, .8, .1]]).requires_grad_()
        t = log([[.8, .15, .05], [.6, .3, .1], [.8, .15, .05], [.4, .35, .25]])
        reference, stats = token_terms(s, t, .5, "token_jsd", False, [2], CLIP)
        actual, _ = token_terms(s, t, .5, "token_jsd", False, [2], CLIP, normalize=False)
        weight = 1 - stats[:, 0]
        self.assertEqual(stats[:, 1].tolist(), [1., 1., 1., 0.])
        self.assertEqual(stats[:, 8].tolist(), [1., 0., 0., 0.])
        torch.testing.assert_close(actual, weight * reference)
        torch.testing.assert_close(actual[3], stats[3, 5], rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(actual.sum(), s, retain_graph=True)[0],
                                   weight.unsqueeze(-1) * torch.autograd.grad(reference.sum(), s)[0])

    def test_unnormalized_pull_toward_teacher_argmax_vanishes(self):
        for clip, normalized_floor in ((INF, .5), (CLIP, .13)):
            z = log([[1e-3, 1 - 1e-3]]).requires_grad_()
            t = log([[.8, .2]])
            pulls = []
            for normalize in (True, False):
                loss, _ = token_terms(z, t, .5, "token_jsd", False, [1], clip, normalize=normalize)
                pulls.append(-torch.autograd.grad(loss.sum(), z)[0][0, 0].item())
            self.assertGreater(pulls[0], normalized_floor)
            self.assertLess(pulls[1], 1e-3)

    def test_dense_loss_forwards_ablation_options(self):
        s = log([[[.1, .9], [.7, .3]], [[.15, .85], [.4, .6]]])
        t = log([[[.8, .2], [.8, .2]], [[.8, .2], [.45, .55]]])
        mask = torch.tensor([[1, 0], [1, 1]])
        s[0, 1] = t[0, 1] = float("nan")
        s.requires_grad_()
        loss, metrics = dense_token_gate_loss(s, t, mask, tau_q=.5, kind="token_jsd", split_stop=False,
                                              stop_ids=[1], clip=INF, token_chunk_size=1, fixed_beta=1.)
        expected = (kl(t[0, 0], s[0, 0]) + kl(t[1], s[1]).mean()) / 2
        torch.testing.assert_close(loss, expected)
        self.assertEqual(metrics["gate_fraction"], 1)
        self.assertEqual(metrics["clip_fraction"], 0)
        loss.backward()
        torch.testing.assert_close(s.grad[0, 1], torch.zeros(2, dtype=torch.float64))

    def test_ablation_validation_rejects_mixed_or_invalid_settings(self):
        validate_ablation("token_jsd", None, True, None, True, True, .1, True)
        validate_ablation("token_mix", .5, True, INF, False, False, None, False)
        for args in (("token_jsd", .5, True, None, False, False, None, False),
                     ("token_jsd", 1.5, True, INF, False, False, None, False),
                     ("token_jsd", True, True, INF, False, False, None, False),
                     ("token_jsd", None, "no", CLIP, False, False, None, False),
                     ("token_mix", None, False, CLIP, False, False, None, False),
                     ("token_jsd", .5, False, INF, False, False, None, False),
                     ("token_jsd", None, False, CLIP, True, False, None, False),
                     ("token_jsd", None, False, CLIP, False, True, None, False),
                     ("token_jsd", None, False, CLIP, False, False, .1, False),
                     ("token_jsd", .5, True, INF, False, False, None, True)):
            with self.assertRaises(ValueError):
                validate_ablation(*args)

    def test_launcher_rejects_ablation_options_outside_token_gate(self):
        with self.assertRaises(ValueError):
            load_config("vOPD", ROOT / "configs/deepseek_token_gate/04_jsd_q05.yaml",
                        ["training.token_gate_beta=0.5"])
        with self.assertRaises(ValueError):
            load_config("vOPD", ROOT / "configs/deepseek_token_gate/07_rkl.yaml",
                        ["training.token_gate_normalize=false"])

    def test_configs_change_only_the_ablated_keys(self):
        baseline = OmegaConf.to_container(load_config("vOPD", ROOT / "configs/deepseek_token_gate/04_jsd_q05.yaml"))
        keys = ("dense_objective", "token_gate_clip", "token_gate_beta", "token_gate_normalize")
        for name, expected in CONFIGS.items():
            config = load_config("vOPD", ROOT / f"configs/deepseek_core_ablation/{name}.yaml")
            validate_assets(config, "vOPD")
            self.assertEqual(config.run.dataset, f"dapo14k_deepseek_core_{name}")
            self.assertEqual(tuple(config.training[k] for k in keys), expected)
            probe, reference = OmegaConf.to_container(config), dict(baseline)
            for item in (probe, reference):
                item["run"] = {k: v for k, v in item["run"].items() if k != "dataset"}
                item["training"] = {k: v for k, v in item["training"].items() if k not in keys}
            self.assertEqual(probe, reference)
        script = (ROOT / "configs/deepseek_core_ablation/run_all.sh").read_text()
        self.assertIn(f"for ablation_name in {' '.join(CONFIGS)}; do", script)


if __name__ == "__main__":
    unittest.main()
