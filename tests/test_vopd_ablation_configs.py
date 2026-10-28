"""Ensure the five ablations differ only in their divergence schedule."""
from pathlib import Path
import unittest
from omegaconf import OmegaConf
from scripts._runtime import load_config

ROOT = Path(__file__).resolve().parents[1]


class AblationConfigs(unittest.TestCase):
    def test_paired_training_and_evaluation(self):
        configs = sorted(p for p in (ROOT/'configs/deepseek_jsd_ablation').glob('*.yaml')
                         if p.name != '00_student.yaml')
        self.assertEqual(len(configs), 5)
        expected = [('teacher_entropy', .5, .5), ('linear', 0., 0.), ('linear', .5, .5),
                    ('teacher_entropy_mean', 0., 1.), ('linear', 0., 1.)]
        shared = None
        for path, schedule in zip(configs, expected):
            c = load_config('vOPD', path)
            self.assertEqual((c.training.alpha_schedule, c.training.alpha_start, c.training.alpha_end), schedule)
            self.assertEqual(c.training.stop_weight, 0)
            self.assertEqual(c.training.max_steps, 211)
            self.assertEqual(c.training.save_steps, 50)
            self.assertIsNone(c.training.save_total_limit)
            self.assertEqual(c.evaluation.seed_mode, 'engine')
            self.assertEqual(c.evaluation.data_parallel_size, 8)
            self.assertEqual(c.training.generation_stop_token_ids, [151643])
            values = OmegaConf.to_container(c, resolve=True)
            values['run'].pop('dataset')
            for key in ['alpha_start', 'alpha_end', 'alpha_schedule']:
                values['training'].pop(key)
            if shared is None:
                shared = values
            else:
                self.assertEqual(values, shared)


if __name__ == '__main__':
    unittest.main()
