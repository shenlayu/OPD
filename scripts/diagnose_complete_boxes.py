"""Rescore saved vOPD outputs using complete boxes, without changing official scores."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from scripts._data import boxed_answer
from scripts._evaluate import grade


def complete_boxes(text):
    position = 0
    while True:
        opening = text.find("\\boxed{", position)
        if opening < 0:
            return
        start = opening + len("\\boxed{")
        depth = 1
        for end in range(start, len(text)):
            depth += (text[end] == "{") - (text[end] == "}")
            if depth == 0:
                yield text[start:end].strip()
                break
        position = start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    source = args.run_dir / "results.jsonl"
    saved = json.loads((args.run_dir / "scores.json").read_text())
    if saved["module"] != "vOPD" or saved["status"] != "complete":
        raise ValueError("This diagnostic requires a completed vOPD evaluation")
    digest = hashlib.file_digest(source.open("rb"), "sha256").hexdigest()
    if digest != saved["results_sha256"]:
        raise ValueError("Generation stream does not match saved scores")

    cache, cases, seen = {}, [], set()
    for line in source.open():
        row = json.loads(line)
        identity = row["benchmark"], row["example_id"], row["sample_id"]
        if identity in seen:
            raise ValueError(f"Duplicate sample: {identity}")
        seen.add(identity)
        if type(row["correct"]) is not bool or row.get("grading_error"):
            raise ValueError(f"Invalid original grade: {identity}")
        old = boxed_answer(row["response"])
        boxes = list(complete_boxes(row["response"]))

        def check(prediction):
            if prediction == old:
                return row["correct"]
            if prediction is None:
                return False
            key = prediction, row["answer"]
            if key not in cache:
                cache[key] = grade(
                    "\\boxed{" + prediction + "}", row["answer"], "vOPD", 10
                )
            return cache[key]

        first = boxes[0] if boxes else None
        last = boxes[-1] if boxes else None
        cases.append(dict(
            benchmark=row["benchmark"], example_id=row["example_id"],
            sample_id=row["sample_id"], truncated=row["finish_reason"] == "length",
            original_correct=row["correct"], original_prediction=old,
            first_complete_correct=check(first), first_complete_prediction=first,
            last_complete_correct=check(last), last_complete_prediction=last,
            complete_box_count=len(boxes),
        ))
        if len(cases) % 1000 == 0:
            print(f"Rescored {len(cases)} responses", flush=True)

    benchmarks = {}
    for name, spec in saved["benchmarks"].items():
        rows = [r for r in cases if r["benchmark"] == name]
        q, n = spec["expected_questions"], spec["samples_per_question"]
        actual = {(r["example_id"], r["sample_id"]) for r in rows}
        expected = {(i, j) for i in range(q) for j in range(n)}
        if actual != expected:
            raise ValueError(f"Incomplete sample coverage: {name}")
        groups = defaultdict(list)
        for row in rows:
            groups[row["example_id"]].append(row)
        metrics = dict(questions=q, samples_per_question=n, samples=len(rows))
        for mode in ("original", "first_complete", "last_complete"):
            correct = sum(r[f"{mode}_correct"] for r in rows)
            passed = sum(any(r[f"{mode}_correct"] for r in rs) for rs in groups.values())
            votes = 0
            for rs in groups.values():
                valid = [r for r in sorted(rs, key=lambda r: r["sample_id"])
                         if r[f"{mode}_prediction"] is not None]
                if valid:
                    winner = Counter(r[f"{mode}_prediction"] for r in valid).most_common(1)[0][0]
                    votes += next(r[f"{mode}_correct"] for r in valid
                                  if r[f"{mode}_prediction"] == winner)
            metrics[mode] = dict(
                correct_samples=correct, avg_at_n_pct=100 * correct / len(rows),
                passed_questions=passed, pass_at_n_pct=100 * passed / q,
                majority_vote_at_n_pct=100 * votes / q,
            )
        if metrics["original"]["correct_samples"] != spec["correct_samples"]:
            raise ValueError(f"Original score mismatch: {name}")
        metrics["rescued_by_last_complete"] = sum(
            r["last_complete_correct"] and not r["original_correct"] for r in rows
        )
        metrics["lost_by_last_complete"] = sum(
            r["original_correct"] and not r["last_complete_correct"] for r in rows
        )
        metrics["truncated_samples"] = sum(r["truncated"] for r in rows)
        metrics["missing_original_but_has_complete_box"] = sum(
            r["original_prediction"] is None and r["last_complete_prediction"] is not None
            for r in rows
        )
        benchmarks[name] = metrics

    result = dict(
        created_at=datetime.now(timezone.utc).isoformat(), status="complete",
        source=str(source.resolve()), source_sha256=digest,
        samples=len(cases), grading_errors=0,
        protocol=("Keep all responses including truncated ones. Choose the first/last "
                  "complete boxed answer without using the gold answer; use the same "
                  "vOPD math verifier. Responses with no complete box remain incorrect. "
                  "Official scores and generation records are unchanged."),
        benchmarks=benchmarks,
    )
    output = args.run_dir / "diagnostics"
    output.mkdir(exist_ok=True)
    (output / "complete_box_scores.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    with (output / "complete_box_cases.jsonl").open("w") as stream:
        for row in cases:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
