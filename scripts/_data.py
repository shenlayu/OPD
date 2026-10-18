"""Local parquet/JSON and downloaded Hub dataset loading for both modules."""

from pathlib import Path


def load_rows(path, split="train"):
    from datasets import load_dataset, load_from_disk

    path = Path(path)
    if path.is_file():
        kind = "parquet" if path.suffix == ".parquet" else "json"
        return load_dataset(kind, data_files={split: str(path)}, split=split)
    if (path / "dataset_info.json").exists() or (path / "dataset_dict.json").exists():
        dataset = load_from_disk(str(path))
        return dataset[split] if hasattr(dataset, "keys") else dataset
    if path.is_dir():
        return load_dataset(str(path), split=split)
    raise FileNotFoundError(
        f"Dataset missing: {path}. See data/README.md for downloads."
    )


def problem_and_answer(row):
    prompt = row.get("prompt")
    if prompt is not None and not isinstance(prompt, str):
        problem = prompt[0]["content"]
    else:
        problem = next(
            (
                row[k]
                for k in ("problem", "question", "Question", "prompt")
                if isinstance(row.get(k), str)
            ),
            None,
        )
    if problem is None:
        raise ValueError(f"No problem text in columns: {list(row)}")
    if isinstance(row.get("reward_model"), dict):
        answer = row["reward_model"]["ground_truth"]
    else:
        answer = next(
            (
                row[k]
                for k in ("answer", "Answer", "solution")
                if row.get(k) is not None
            ),
            None,
        )
    if answer is None:
        raise ValueError("Evaluation dataset has no ground truth")
    return str(problem), str(answer)


def boxed_answer(text):
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    start += len("\\boxed{")
    depth = 1
    for i in range(start, len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            return text[start:i].strip()
    return None
