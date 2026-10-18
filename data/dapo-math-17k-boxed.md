# DAPO-Math-17k boxed 派生训练集

原目录还使用过从原始 DAPO 去重、改为 boxed 答案指令的本地 parquet：
原始 1,791,700 行，去重后 17,917 题。它与作者直接发布的
[rethinking DAPO parquet](dapo-math-17k.md) 是不同的数据制备来源，不应混称为同一配置。
这里仅提供原始资源下载和派生命令，不迁移数据。

```bash
hf download BytedTsinghua-SIA/DAPO-Math-17k data/dapo-math-17k.parquet \
  --repo-type dataset --revision 65877096c24ffa7abc4e4fa5edb95cf3413a5674 \
  --local-dir data/dapo-raw
python - <<'PY'
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

prefix = ('Solve the following math problem step by step. The last line of your response '
          'should be of the form Answer: $Answer (without quotes) where $Answer is the answer '
          'to the problem.\n\n')
suffix = '\n\nRemember to put your answer on its own line after "Answer:".'
seen, rows = set(), []
source = pq.ParquetFile('data/dapo-raw/data/dapo-math-17k.parquet')
for batch in source.iter_batches(batch_size=4096):
    for row in batch.to_pylist():
        text = row['prompt'][0]['content']
        assert text.startswith(prefix) and text.endswith(suffix)
        question = text[len(prefix):-len(suffix)]
        if question in seen:
            continue
        seen.add(question)
        row['prompt'] = [{'role': 'user', 'content': question +
            '\n\nPlease reason step by step, and put your final answer within \\boxed{}.'}]
        row['ability'] = 'math'
        row['reward_model']['style'] = 'rule'
        rows.append(row)
assert len(rows) == 17917, len(rows)
Path('data/rethinking').mkdir(parents=True, exist_ok=True)
pq.write_table(pa.Table.from_pylist(rows), 'data/rethinking/dapo-math-17k-boxed.parquet')
PY
```

配置 `training.data.train_files: [data/rethinking/dapo-math-17k-boxed.parquet]`。
原始文件 SHA256：`534375d6bb8630d22ab46a56e11f2ffec1d288d8f7d04099bc82d68948705941`。
