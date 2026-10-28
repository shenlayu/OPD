import unittest
import torch
from modules.vOPD.src.progressive_jsd import dense_jsd_loss, stop_bernoulli_kl


class StopAuxiliaryTests(unittest.TestCase):
    def test_matches_grouped_bernoulli_value_and_gradient(self):
        q = torch.tensor([[0.08, 0.3, 0.12, 0.5]], dtype=torch.float64).log().requires_grad_()
        p = torch.tensor([[0.35, 0.2, 0.25, 0.2]], dtype=torch.float64).log().requires_grad_()
        actual, ps, qs = stop_bernoulli_kl(q.log_softmax(-1), p.log_softmax(-1), [0, 2])
        expected = torch.distributions.kl_divergence(
            torch.distributions.Bernoulli(probs=torch.tensor([0.6], dtype=torch.float64)),
            torch.distributions.Bernoulli(probs=q.softmax(-1)[:, [0, 2]].sum(-1)),
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual.sum(), q, retain_graph=True)[0],
                                   torch.autograd.grad(expected.sum(), q)[0])
        self.assertIsNone(torch.autograd.grad(actual.sum(), p, allow_unused=True)[0])
        torch.testing.assert_close(ps, torch.tensor([0.6], dtype=torch.float64))
        torch.testing.assert_close(qs, torch.tensor([0.2], dtype=torch.float64))

    def test_mask_chunking_and_zero_weight(self):
        torch.manual_seed(42)
        q = torch.randn(2, 3, 7, dtype=torch.float64, requires_grad=True)
        p = torch.randn_like(q)
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        qlog, plog = q.log_softmax(-1), p.log_softmax(-1)
        rkl = (qlog.exp() * (qlog - plog)).sum(-1)
        ps, qs = plog.exp()[..., [0, 6]].sum(-1), qlog.exp()[..., [0, 6]].sum(-1)
        stop = ps * (ps.log() - qs.log()) + (1-ps) * ((1-ps).log() - (1-qs).log())
        expected = ((rkl + stop) * mask).sum() / mask.sum()
        value, stats = dense_jsd_loss(q, p, mask, 1, token_chunk_size=2,
                                     stop_weight=1, stop_token_ids=[0, 6], return_alpha_stats=True)
        torch.testing.assert_close(value, expected)
        grad = torch.autograd.grad(value, q, retain_graph=True)[0]
        torch.testing.assert_close(grad, torch.autograd.grad(expected, q, retain_graph=True)[0])
        self.assertEqual(float(grad[mask == 0].abs().sum()), 0)
        torch.testing.assert_close(value, stats['primary_loss'] + stats['stop_loss'])
        base = dense_jsd_loss(q, p, mask, 1, stop_weight=0)
        torch.testing.assert_close(base, (rkl * mask).sum() / mask.sum())

    def test_extreme_stop_probabilities_have_finite_gradients(self):
        q = torch.tensor([[1000., -1000., -1001.], [-1000., 1000., 999.]], requires_grad=True)
        p = -q.detach()
        loss, _, _ = stop_bernoulli_kl(q.log_softmax(-1), p.log_softmax(-1), [0])
        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue(torch.isfinite(torch.autograd.grad(loss.sum(), q)[0]).all())

    def test_invalid_stop_sets(self):
        q = torch.zeros(2, 3).log_softmax(-1)
        for ids in [[], [1, 1], [-1], [3], [0, 1, 2]]:
            with self.assertRaises(ValueError):
                stop_bernoulli_kl(q, q, ids)


if __name__ == '__main__':
    unittest.main()
