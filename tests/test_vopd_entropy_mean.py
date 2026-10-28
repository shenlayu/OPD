"""CPU checks of token-weighted aggregation over a whole optimizer batch."""

from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from trl.experimental.minillm import MiniLLMTrainer
from trl.trainer.utils import shuffle_sequence_dict, split_tensor_dict

from modules.vOPD.src.opd_trainer import CustomMiniLLMTrainer
from modules.vOPD.src.progressive_jsd import entropy_alpha, entropy_alpha_totals, token_divergence


class Teacher(torch.nn.Module):
    def forward(self, input_ids, attention_mask, position_ids, use_cache):
        expected = attention_mask.long().cumsum(-1) - 1
        expected.masked_fill_(attention_mask == 0, 0)
        torch.testing.assert_close(position_ids, expected)
        x = input_ids.double() / 2
        return NS(logits=torch.stack((x, -x, torch.zeros_like(x)), dim=-1))


def batch(rank=0):
    return {
        "prompt_ids": torch.tensor([[0, 1 + 6 * rank], [2, 3 + 6 * rank]]),
        "prompt_mask": torch.tensor([[0, 1], [1, 1]]),
        "completion_ids": torch.tensor([[2 + rank, 3, 0], [4 + 4 * rank, 1, 2]]),
        "completion_mask": (torch.tensor([[1, 1, 0], [1, 0, 0]]) if rank == 0
                            else torch.tensor([[1, 1, 1], [1, 1, 1]])),
    }


def trainer():
    t = object.__new__(CustomMiniLLMTrainer)
    t.args = NS(per_device_train_batch_size=1)
    t.teacher_model = Teacher()
    t.kd_temperature = 1.3
    t.loss_token_chunk_size = 2
    t.dense_objective = "progressive_jsd"
    t.alpha_schedule = "teacher_entropy_mean"
    t.alpha_start, t.alpha_end = 0., 1.
    t.state = NS(global_step=3, max_steps=11)
    t.model = NS(training=True)
    t.stop_weight, t.stop_token_ids = 0., ()
    t._metrics = {"train": defaultdict(list)}
    return t


def response_logits(t, inputs):
    ids = torch.cat((inputs["prompt_ids"], inputs["completion_ids"]), dim=1)
    mask = torch.cat((inputs["prompt_mask"], inputs["completion_mask"]), dim=1)
    positions = mask.long().cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    return t.teacher_model(ids, mask, positions, False).logits[:, inputs["prompt_ids"].shape[1]-1:-1]


def reference_totals(t, inputs, progress):
    logits = response_logits(t, inputs) / t.kd_temperature
    alphas = entropy_alpha(logits.log_softmax(-1), progress)
    weights = inputs["completion_mask"].double()
    return torch.stack(((alphas * weights).sum(), weights.sum()))


def distributed_check(rank, rendezvous):
    distributed.init_process_group("gloo", init_method=f"file://{rendezvous}",
                                   rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        t = trainer()
        actual = t._aggregate_entropy_alpha(batch(rank), .3)
        refs = [reference_totals(t, batch(r), .3) for r in range(2)]
        total = sum(refs)
        expected = (total[0] / total[1]).item()
        assert abs(actual - expected) < 1e-12, (rank, actual, expected)
        wrong = sum((r[0] / r[1]).item() for r in refs) / 2
        assert abs(expected - wrong) > 1e-3, "Fixture must distinguish rank mean from token mean"
    finally:
        distributed.destroy_process_group()


class EntropyMeanTests(unittest.TestCase):
    def test_transform_before_average_and_ignore_masked_tokens(self):
        logits = torch.tensor([[[0., 0., 0.], [9., 0., -9.], [float('nan')]*3]], requires_grad=True)
        mask = torch.tensor([[1, 1, 0]])
        log_p = logits.detach()[0, :2].log_softmax(-1)
        expected = entropy_alpha(log_p, .4).double().sum()
        for chunk_size in [1, 2, 8]:
            totals = entropy_alpha_totals(logits, mask, .4, chunk_size)
            torch.testing.assert_close(totals, torch.stack((expected, expected.new_tensor(2))))
            self.assertFalse(totals.requires_grad)
        h_mean = (-(log_p.exp() * log_p).sum(-1) / torch.log(torch.tensor(2.))).mean()
        self.assertGreater(abs(expected.item()/2 - .4**(1+h_mean.item())), .01)

    def test_aggregate_all_accumulation_slices(self):
        t, inputs = trainer(), batch()
        sums = reference_totals(t, inputs, .3)
        rng_before = torch.random.get_rng_state()
        actual = t._aggregate_entropy_alpha(inputs, .3)
        self.assertAlmostEqual(actual, (sums[0] / sums[1]).item(), places=12)
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        t.args.per_device_train_batch_size = 2
        self.assertAlmostEqual(t._aggregate_entropy_alpha(inputs, .3), actual, places=12)

    def test_shared_alpha_survives_rollout_shuffle_and_accumulation(self):
        t, inputs = trainer(), batch()
        with patch.object(MiniLLMTrainer, "_generate_and_score_completions", return_value=inputs):
            result = t._generate_and_score_completions([])
        expected = t._aggregate_entropy_alpha(inputs, .3)
        slices = split_tensor_dict(shuffle_sequence_dict(result), 2)
        for part in slices:
            self.assertEqual(part["distill_shared_alpha"].item(), expected)
        self.assertEqual(sum(p["completion_mask"].sum().item() for p in slices), 3)

    def test_trl_buffer_reuses_alpha_then_refreshes_next_update(self):
        t = trainer()
        t.args.steps_per_generation, t.num_iterations = 2, 1
        t.args.report_to = []
        t.accelerator = NS(is_main_process=True)
        t._step, t._buffered_inputs = 0, None
        with patch.object(MiniLLMTrainer, "_generate_and_score_completions", side_effect=lambda _: batch()) as generate:
            first = t._prepare_inputs([])
            t._step = 1
            second = t._prepare_inputs([])
            self.assertEqual(generate.call_count, 1)
            self.assertEqual(first['distill_shared_alpha'].item(), second['distill_shared_alpha'].item())
            self.assertEqual(first['prompt_ids'].shape[0], 1)
            t._step, t.state.global_step = 2, 4
            third = t._prepare_inputs([])
            self.assertEqual(generate.call_count, 2)
            self.assertGreater(third['distill_shared_alpha'].item(), first['distill_shared_alpha'].item())

    def test_dense_loss_uses_shared_alpha_and_keeps_student_gradient(self):
        t, inputs = trainer(), batch()
        shared = t._aggregate_entropy_alpha(inputs, .3)
        inputs["distill_shared_alpha"] = torch.full((2,), shared, dtype=torch.float64)
        class Student(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.arange(30, dtype=torch.float64).reshape(2, 5, 3) / 7)
            def forward(self, **kwargs):
                return NS(logits=self.logits)
        student = Student()
        actual = t.compute_dense_loss(student, inputs)
        q = student.logits[:, 1:-1] / t.kd_temperature
        p = response_logits(t, inputs) / t.kd_temperature
        values = token_divergence(q, p, shared)
        mask = inputs["completion_mask"]
        expected = (values * mask).sum() / mask.sum()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, student.logits, retain_graph=True)[0],
                                   torch.autograd.grad(expected, student.logits)[0])
        self.assertAlmostEqual(t._metrics['train']['distill/alpha_progress'][0], .3)
        self.assertAlmostEqual(t._metrics['train']['distill/alpha'][0], shared, places=6)

    def test_exact_endpoints_skip_teacher_pass(self):
        t = trainer()
        with patch.object(t.teacher_model, 'forward', side_effect=AssertionError('Unneeded teacher forward')):
            self.assertEqual(t._aggregate_entropy_alpha(batch(), 0), 0)
            self.assertEqual(t._aggregate_entropy_alpha(batch(), 1), 1)

    def test_empty_response_batch_is_rejected(self):
        t, inputs = trainer(), batch()
        inputs['completion_mask'].zero_()
        with self.assertRaisesRegex(ValueError, 'no valid response'):
            t._aggregate_entropy_alpha(inputs, .3)

    def test_unequal_token_counts_across_two_ranks(self):
        with TemporaryDirectory() as tmp:
            multiprocessing.start_processes(distributed_check, args=(str(Path(tmp)/'gloo'),),
                                            nprocs=2, start_method='spawn', join=True)


if __name__ == '__main__':
    unittest.main()
