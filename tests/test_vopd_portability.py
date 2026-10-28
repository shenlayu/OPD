"""Check relocation and the student-only evaluation lifecycle without GPUs."""

from contextlib import nullcontext
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
from scripts._runtime import load_config, validate_assets
import scripts._runtime as runtime
import scripts.evaluate_vopd_student as student

ROOT = Path(__file__).resolve().parents[1]


class PortabilityTests(unittest.TestCase):
    def test_all_six_configs_have_only_repo_relative_paths(self):
        paths = sorted((ROOT/'configs/deepseek_jsd_ablation').glob('*.yaml'))
        self.assertEqual(len(paths), 6)
        for path in paths:
            config = OmegaConf.load(path)
            self.assertIsNone(config.run.python)
            values = [config.run.output_root, config.training.dataset_name,
                      config.training.model_name_or_path, config.training.teacher_model_name_or_path]
            values += [task.path for task in config.evaluation.datasets]
            for value in filter(None, values):
                self.assertFalse(Path(value).is_absolute(), (path.name, value))
                self.assertNotIn('..', Path(value).parts)

    def fixture(self, destination):
        cfg = ROOT/'configs/deepseek_jsd_ablation/00_student.yaml'
        path = destination/'configs/deepseek_jsd_ablation/00_student.yaml'
        path.parent.mkdir(parents=True)
        shutil.copy2(cfg, path)
        model = destination/'model/DeepSeek-R1-Distill-Qwen-1.5B'
        model.mkdir(parents=True)
        for name in ['config.json', 'tokenizer_config.json']:
            (model/name).write_text('{}')
        (model/'model.safetensors').write_bytes(b'fixture, not real weights')
        for name in ['math500', 'minerva', 'amc23', 'aime24', 'aime25']:
            data = destination/'data'/name/'test.parquet'
            data.parent.mkdir(parents=True)
            data.write_bytes(b'fixture, not read by asset validation')
        return path, load_config('vOPD', path, root=destination)

    def test_relocated_student_requires_no_teacher_or_training_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'relocated OPD'
            _, config = self.fixture(root)
            validate_assets(config, 'vOPD', evaluation_only=True)
            self.assertEqual(config.training.model_name_or_path, str(root/'model/DeepSeek-R1-Distill-Qwen-1.5B'))
            self.assertFalse(config.training.use_peft)
            self.assertIsNone(config.training.teacher_model_name_or_path)
            self.assertIsNone(config.training.dataset_name)
            self.assertTrue(all(Path(t.path).is_relative_to(root) for t in config.evaluation.datasets))

    def test_student_runs_evaluation_then_scoring_and_never_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            requested, config = self.fixture(Path(tmp)/'relocated OPD')
            def snapshot(_root, target, _module):
                target.mkdir(parents=True)
            with patch.object(student, 'reserve_gpus', return_value=nullcontext()), \
                 patch.object(student, 'snapshot_code', side_effect=snapshot), \
                 patch.object(student, 'run_command') as execute:
                run = student.evaluate_student(config, requested)
            commands = [c.args[0] for c in execute.call_args_list]
            self.assertEqual([c[2] for c in commands], ['scripts._evaluate', 'scripts.diagnose_complete_boxes'])
            self.assertEqual(commands[0][5], config.training.model_name_or_path)
            self.assertFalse((run/'checkpoint').exists())
            saved = OmegaConf.load(run/'configs/vopd.yaml')
            self.assertFalse(saved.training.use_peft)
            self.assertEqual(json.loads((run/'launch.json').read_text())['status'], 'completed')

    def test_student_failure_stops_before_rescoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            requested, config = self.fixture(Path(tmp)/'relocated OPD')
            def snapshot(_root, target, _module):
                target.mkdir(parents=True)
            with patch.object(student, 'reserve_gpus', return_value=nullcontext()), \
                 patch.object(student, 'snapshot_code', side_effect=snapshot), \
                 patch.object(student, 'run_command', side_effect=RuntimeError('evaluation failed')) as execute:
                with self.assertRaisesRegex(RuntimeError, 'evaluation failed'):
                    student.evaluate_student(config, requested)
            self.assertEqual(execute.call_count, 1)
            records = list(Path(config.run.output_root).rglob('launch.json'))
            self.assertEqual(json.loads(records[0].read_text())['status'], 'failed')

    def test_training_launcher_automatically_evaluates_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'relocated OPD'
            requested, config = self.fixture(root)
            training_data = root/'data/train.parquet'
            training_data.write_bytes(b'fixture')
            config.training.dataset_name = str(training_data)
            config.training.teacher_model_name_or_path = config.training.model_name_or_path
            config.training.use_peft = True
            OmegaConf.save(config, requested)
            def snapshot(_root, target, _module):
                target.mkdir(parents=True)
            argv = ['run_vopd.py', '--config', str(requested)]
            with patch('sys.argv', argv), patch.object(runtime, 'snapshot_code', side_effect=snapshot), \
                 patch.object(runtime, 'run_command') as execute:
                self.assertEqual(runtime.launch('vOPD', 'vopd.yaml'), 0)
            commands = [c.args[0] for c in execute.call_args_list]
            self.assertIn('modules.vOPD.src.opd_train', commands[0])
            self.assertEqual([c[2] for c in commands[1:]], ['scripts._evaluate', 'scripts.diagnose_complete_boxes'])
            self.assertTrue(commands[1][5].endswith('/checkpoint/final'))
            with patch('sys.argv', argv), patch.object(runtime, 'snapshot_code', side_effect=snapshot), \
                 patch.object(runtime, 'run_command', side_effect=RuntimeError('training failed')) as execute:
                with self.assertRaisesRegex(RuntimeError, 'training failed'):
                    runtime.launch('vOPD', 'vopd.yaml')
            self.assertEqual(execute.call_count, 1)


if __name__ == '__main__':
    unittest.main()
