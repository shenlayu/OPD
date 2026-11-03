"""Check objective values, gradients, masked normalization and seven-run protocol."""

from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from modules.vOPD.src.token_gate import token_terms, dense_token_gate_loss, calibrate_clip
from scripts._runtime import load_config, validate_assets
from scripts._parallel_evaluate import validate_parallel_config

ROOT = Path(__file__).resolve().parents[1]


def check_distributed_clip(rank, filename):
    dist.init_process_group('gloo', init_method=f'file://{filename}', rank=rank, world_size=2,
                            timeout=timedelta(seconds=40))
    try:
        s = torch.tensor([[[0., 1., -1.], [-2., 0., 1.]]]) + rank * torch.tensor([1., 0., -1.])
        t = torch.tensor([[[3., 0., -1.], [0., 3., -1.]]])
        mask = torch.tensor([[1, rank]])
        value, count = calibrate_clip(s, t, mask, .5, False, [2], 1)
        refs = []
        for r in [0, 1]:
            ss = torch.tensor([[0., 1., -1.], [-2., 0., 1.]]) + r * torch.tensor([1., 0., -1.])
            _, stats = token_terms(ss[:1+r], t[0, :1+r], .5, 'token_mix', False, [2])
            refs.append(stats[:, 0] * stats[:, 4])
        expected = torch.quantile(torch.cat(refs), .95).item()
        assert count == 3 and abs(value - expected) < 1e-6, (value, expected, count)
    finally:
        dist.destroy_process_group()


class TokenGateTests(unittest.TestCase):
    def setUp(self):
        self.s = torch.tensor([[.2, -.3, .7], [-2., 1., -.5]], dtype=torch.float64, requires_grad=True)
        self.t = torch.tensor([[3., 0., -1.], [.1, .2, .3]], dtype=torch.float64, requires_grad=True)

    def test_mix_value_and_gradient_with_detached_gate(self):
        actual, stats = token_terms(self.s, self.t, .5, 'token_mix', False, [2])
        lp, lq = self.s.log_softmax(-1), self.t.detach().log_softmax(-1)
        b = stats[:, 0]
        expected = b * (lq.exp() * (lq-lp)).sum(-1) + (1-b) * (lp.exp() * (lp-lq)).sum(-1)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual.sum(), self.s, retain_graph=True)[0],
                                   torch.autograd.grad(expected.sum(), self.s)[0])
        self.assertEqual(stats[1, 0], 0)
        self.assertIsNone(torch.autograd.grad(actual.sum(), self.t, allow_unused=True)[0])

    def test_jsd_reference_and_exact_rkl_gate_off(self):
        actual, stats = token_terms(self.s, self.t, .5, 'token_jsd', False, [2])
        lp, lq = self.s.log_softmax(-1), self.t.detach().log_softmax(-1)
        b = stats[0, 0]
        m = (1-b)*lq[0].exp() + b*lp[0].exp()
        expected = (lq[0].exp()*(lq[0]-m.log())).sum()/b + (lp[0].exp()*(lp[0]-m.log())).sum()/(1-b)
        torch.testing.assert_close(actual[0], expected)
        torch.testing.assert_close(actual[1], (lp[1].exp()*(lp[1]-lq[1])).sum())
        torch.testing.assert_close(torch.autograd.grad(actual[0], self.s, retain_graph=True)[0],
                                   torch.autograd.grad(expected, self.s)[0])

    def test_stop_logit_gradient_is_probability_difference_without_content_leak(self):
        for kind in ['token_mix', 'token_jsd']:
            shift = torch.tensor(.4, dtype=torch.float64, requires_grad=True)
            s = self.s.detach() + shift * torch.tensor([0., 1., 1.])
            loss, stats = token_terms(s, self.t, .5, kind, True, [1, 2], clip=100.)
            gradient = torch.autograd.grad(loss.sum(), shift)[0]
            expected = (s.softmax(-1)[:, 1:].sum(-1) - self.t.detach().softmax(-1)[:, 1:].sum(-1)).sum()
            torch.testing.assert_close(gradient, expected)
            self.assertTrue(torch.isfinite(stats).all())

    def test_sequence_means_mask_nan_padding_and_chunk_gradients(self):
        s = self.s.detach().repeat(2, 2, 1).requires_grad_()
        t = self.t.detach().repeat(2, 2, 1)
        mask = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 0]])
        s.data[~mask.bool()] = float('nan')
        t[~mask.bool()] = float('nan')
        for kind in ['token_mix', 'token_jsd']:
            loss, _ = dense_token_gate_loss(s, t, mask, tau_q=.5, kind=kind, split_stop=True,
                                            stop_ids=[2], clip=2., token_chunk_size=2)
            values0, _ = token_terms(s[0, :1], t[0, :1], .5, kind, True, [2], 2.)
            values1, _ = token_terms(s[1, :3], t[1, :3], .5, kind, True, [2], 2.)
            expected = (values0.mean() + values1.mean()) / 2
            torch.testing.assert_close(loss, expected)
            a = torch.autograd.grad(loss, s, retain_graph=True)[0]
            b = torch.autograd.grad(expected, s, retain_graph=True)[0]
            torch.testing.assert_close(a, b)
            self.assertTrue(torch.isfinite(a).all())
            self.assertTrue((a[~mask.bool()] == 0).all())

    def test_clip_blocks_only_forward_gradient(self):
        loss, stats = token_terms(self.s, self.t, .5, 'token_mix', False, [2], .001)
        b = stats[:, 0]
        lp, lq = self.s.log_softmax(-1), self.t.detach().log_softmax(-1)
        expected = .001 + (1-b[0]) * (lp[0].exp() * (lp[0]-lq[0])).sum()
        self.assertEqual(stats[0, 8], 1)
        torch.testing.assert_close(loss[0], expected)
        torch.testing.assert_close(torch.autograd.grad(loss[0], self.s, retain_graph=True)[0],
                                   torch.autograd.grad(expected, self.s)[0])

    def test_extreme_probabilities_have_finite_values_and_gradients(self):
        for kind in ['token_mix', 'token_jsd']:
            for split in [False, True]:
                s = torch.tensor([[1000., -1000., -1001.], [-1000., 1000., 999.]], requires_grad=True)
                loss, stats = token_terms(s, -s.detach(), .5, kind, split, [0], 5.)
                self.assertTrue(torch.isfinite(loss).all())
                self.assertTrue(torch.isfinite(stats).all())
                self.assertTrue(torch.isfinite(torch.autograd.grad(loss.sum(), s)[0]).all())

    def test_calibration_gathers_unequal_valid_counts_across_ranks(self):
        with TemporaryDirectory() as temp:
            mp.start_processes(check_distributed_clip, args=(str(Path(temp)/'gloo'),),
                               nprocs=2, start_method='spawn', join=True)

    def test_seven_configs_only_change_intended_factors(self):
        paths = sorted((ROOT / 'configs/deepseek_token_gate').glob('*.yaml'))
        self.assertEqual(len(paths), 7)
        expected = [('token_mix', .5, False), ('token_mix', .7, False), ('token_mix', .5, True),
                    ('token_jsd', .5, False), ('token_jsd', .7, False), ('token_jsd', .5, True),
                    ('reverse_kl', .5, False)]
        reference = None
        for path, variant in zip(paths, expected):
            c = load_config('vOPD', path)
            validate_assets(c, 'vOPD')
            self.assertEqual((c.training.dense_objective, c.training.token_gate_tau_q, c.training.token_gate_stop), variant)
            self.assertEqual(c.training.stop_weight, 0)
            self.assertEqual(c.training.max_steps, 211)
            self.assertEqual(len(c.evaluation.datasets), 5)
            self.assertTrue(all(x.samples == 4 and x.limit is None for x in c.evaluation.datasets))
            self.assertEqual(c.evaluation.max_new_tokens, 8192)
            self.assertEqual(c.training.generation_stop_token_ids, [151643])
            data = OmegaConf.to_container(c)
            data['run'].pop('dataset')
            for key in ['dense_objective', 'token_gate_tau_q', 'token_gate_stop']:
                data['training'].pop(key)
            if reference is None:
                reference = data
            else:
                self.assertEqual(data, reference)

    def test_scheduler_uuid_mapping_reaches_evaluation_workers(self):
        config = OmegaConf.create(dict(run=dict(gpus=[3, 1]), evaluation=dict(
            data_parallel_size=2, tensor_parallel_size=1, seed_mode='engine')))
        env = dict(CUDA_VISIBLE_DEVICES='GPU-three,GPU-one', GPU_SCHEDULE_GPU_IDS='3,1')
        with patch.dict('os.environ', env):
            self.assertEqual(validate_parallel_config(config, 'vOPD'), ['GPU-three', 'GPU-one'])
            config.run.gpus = [0, 1]
            with self.assertRaisesRegex(ValueError, 'outside GPU scheduler'):
                validate_parallel_config(config, 'vOPD')


if __name__ == '__main__':
    unittest.main()
