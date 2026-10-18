# amc23

资源占位，未迁移数据。来源：[AI-MO/aimo-validation-amc](https://huggingface.co/datasets/AI-MO/aimo-validation-amc)。

在 OPD_lite 根目录执行：

```bash
hf download AI-MO/aimo-validation-amc --repo-type dataset \
  --revision 69d78a4a2c840e82d69af6bc742bda09005f6316 --local-dir data/amc23/raw
python - <<'PY'
from pathlib import Path
from datasets import load_dataset
files = sorted(str(p) for p in Path('data/amc23/raw').rglob('*.parquet'))
kind = 'parquet'
assert files, 'No benchmark files downloaded'
rows = load_dataset(kind, data_files=files, split='train')
rows.to_parquet('data/amc23/test.parquet')
PY
```

原始来源 split：`train`；本地统一命名 `test.parquet`，不改变题目和答案。

本项目采用 AI-MO 这一版本，保留原题目和答案；不要混用另一个 AMC23 发布版做分数比较。
