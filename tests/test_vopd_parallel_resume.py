"""CPU checks for complete-question engine sharding and partial-result reuse."""
import unittest
from types import SimpleNamespace as NS
from omegaconf import OmegaConf
import scripts._evaluate as evaluate


class Tokenizer:
    eos_token_id = 2
    all_special_ids = [2]
    def encode(self, text, **kwargs):
        return [2]


class Engine:
    def __init__(self):
        self.requests = []
    def generate(self, prompts, params, **kwargs):
        self.requests.extend(zip(prompts, params))
        return [NS(outputs=[NS(text=r'\boxed{1}', token_ids=[1], finish_reason='stop', stop_reason=2)
                            for _ in range(p['n'])]) for p in params]


class Writer:
    def __init__(self):
        self.rows = []
    def write(self, row):
        self.rows.append(row)
    def flush(self):
        pass


class ShardTest(unittest.TestCase):
    def setUp(self):
        self.old_grade = evaluate.grade
        evaluate.grade = lambda *args: True
        self.config = OmegaConf.create({'evaluation': dict(
            seed_mode='engine', seed=42, seed_task_stride=100000, seed_example_stride=32,
            prompt_style='plain_math', temperature=.6, top_p=.9, top_k=-1, min_p=0,
            presence_penalty=0, repetition_penalty=1, max_new_tokens=4096,
            stop_mode='registered_special', batch_size=2, max_prompt_length=1024, grading_timeout=10,
        )})
        self.task = OmegaConf.create({'name':'bench', 'samples':8})
        self.rows = [{'problem':str(i),'answer':'1'} for i in range(11)]
    def tearDown(self):
        evaluate.grade = self.old_grade
    def run_shards(self, skip=frozenset(), size=4):
        all_rows, engines = [], []
        for rank in range(size):
            writer, engine = Writer(), Engine()
            evaluate.evaluate_tasks(self.config, 'vOPD', Tokenizer(), engine, lambda **kw:kw,
                                    writer, [(self.task,self.rows)], shard_rank=rank,
                                    shard_world_size=size, skip_samples=skip)
            all_rows.extend(writer.rows);engines.append(engine)
            if self.config.evaluation.seed_mode == 'engine':
                self.assertTrue(all(r['example_id'] % size == rank for r in writer.rows))
        return all_rows, engines
    def test_engine_group_coverage_without_duplicates(self):
        rows, engines = self.run_shards()
        self.assertEqual(len(rows),88)
        self.assertEqual(len({(r['example_id'],r['sample_id']) for r in rows}),88)
        self.assertTrue(all(r['seed'] is None for r in rows))
        self.assertTrue(all(p['n']==8 and p['seed'] is None for e in engines for _,p in e.requests))
    def test_eight_engine_question_shards_cover_all_samples(self):
        from scripts._parallel_evaluate import validate_parallel_config
        self.config.run = {'gpus': list(range(8))}
        self.config.evaluation.data_parallel_size = 8
        self.config.evaluation.tensor_parallel_size = 1
        self.assertEqual(validate_parallel_config(self.config, 'vOPD'), list(map(str, range(8))))
        with self.assertRaises(ValueError):
            validate_parallel_config(self.config, 'rethinking')
        rows, engines = self.run_shards(size=8)
        self.assertEqual(len(rows), 88)
        self.assertEqual(len({(r['example_id'],r['sample_id']) for r in rows}), 88)
        self.assertTrue(all(e.requests for e in engines))
    def test_preserve_complete_and_partial_groups(self):
        skip = {('bench',0,s) for s in range(8)} | {('bench',1,0),('bench',1,1)}
        rows, engines = self.run_shards(skip)
        self.assertEqual(len(rows),78)
        self.assertTrue(not skip.intersection((r['benchmark'],r['example_id'],r['sample_id']) for r in rows))
        self.assertEqual(sum(len(e.requests) for e in engines),10)
    def test_per_sample_seed_and_coverage_unchanged(self):
        self.config.evaluation.seed_mode='per_sample'
        rows, engines = self.run_shards({('bench',2,3)})
        self.assertEqual(len(rows),87)
        self.assertEqual(len({(r['example_id'],r['sample_id']) for r in rows}),87)
        for r in rows:
            self.assertEqual(r['seed'],42+r['example_id']*32+r['sample_id'])
        self.assertTrue(all(p['n']==1 for e in engines for _,p in e.requests))


if __name__ == '__main__':
    unittest.main()
