"""Check entropy scheduling and its analytic student-logit gradient."""

import unittest
import torch
from modules.vOPD.src.progressive_jsd import entropy_alpha, token_divergence


class EntropyTests(unittest.TestCase):
    def test_bit_entropy_and_exact_endpoints(self):
        log_p = torch.tensor([[0.5, 0.5], [0.99, 0.01]], dtype=torch.float64).log()
        alpha = entropy_alpha(log_p, 0.5)
        self.assertAlmostEqual(alpha[0].item(), 0.25)
        self.assertGreater(alpha[1].item(), alpha[0].item())
        self.assertTrue(torch.equal(entropy_alpha(log_p, 0), torch.zeros(2, dtype=log_p.dtype)))
        self.assertTrue(torch.equal(entropy_alpha(log_p, 1), torch.ones(2, dtype=log_p.dtype)))

    def test_interior_gradient_matches_analytic_normalized_jsd(self):
        q_logits = torch.tensor([[1., 0., -1.], [-2., 1., 0.]], dtype=torch.float64, requires_grad=True)
        p_logits = torch.tensor([[0., 0., 0.], [1., -1., 0.]], dtype=torch.float64, requires_grad=True)
        loss, alpha = token_divergence(q_logits, p_logits, 0.4,
                                      alpha_schedule='teacher_entropy', return_alpha=True)
        q, p = q_logits.softmax(-1), p_logits.detach().softmax(-1)
        a = alpha.unsqueeze(-1)
        m = a * p + (1-a) * q
        log_ratio = q.log() - m.log()
        expected = q / a * (log_ratio - (q * log_ratio).sum(-1, keepdim=True))
        actual, teacher_grad = torch.autograd.grad(loss.sum(), (q_logits, p_logits), allow_unused=True)
        torch.testing.assert_close(actual, expected)
        self.assertIsNone(teacher_grad)
        self.assertFalse(alpha.requires_grad)


if __name__ == '__main__':
    unittest.main()
