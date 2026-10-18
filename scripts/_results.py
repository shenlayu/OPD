"""One score document and one generation stream for an evaluation group."""

from collections import Counter
import hashlib
import json
from pathlib import Path


class ResultWriter:
    """Validate identities and aggregate counts without retaining response text.

    A benchmark gets final percentages only after every expected sample has a
    boolean grade. Exceptions preserve the saved rows and a partial score file.
    Existing results are never silently overwritten.
    """

    def __init__(self, directory, metadata, benchmarks):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(
            (self.directory / name).exists()
            for name in ("scores.json", "results.jsonl")
        ):
            raise FileExistsError(f"Evaluation results already exist: {self.directory}")
        self.metadata = dict(metadata)
        self.specs = benchmarks
        if not benchmarks or any(q < 1 or n < 1 for q, n in benchmarks.values()):
            raise ValueError(
                "Evaluation requires nonempty benchmarks and positive sample counts"
            )
        self.questions = {name: {} for name in benchmarks}
        self.counts = {name: Counter() for name in benchmarks}
        self.digest = hashlib.sha256()
        self.stream = None

    def __enter__(self):
        self.stream = (self.directory / "results.jsonl").open("x", encoding="utf-8")
        try:
            self.flush()
        except BaseException:
            self.stream.close()
            raise
        return self

    def write(self, row):
        task, index, sample = row["benchmark"], row["example_id"], row["sample_id"]
        if task not in self.specs:
            raise ValueError(f"Unexpected benchmark: {task}")
        q, n = self.specs[task]
        if (
            type(index) is not int
            or type(sample) is not int
            or not (0 <= index < q and 0 <= sample < n)
        ):
            raise ValueError(f"Unexpected sample identity: {(task, index, sample)}")
        question = self.questions[task].setdefault(index, {})
        if sample in question:
            raise ValueError(f"Duplicate sample: {(task, index, sample)}")
        correct, error = row["correct"], row.get("grading_error")
        if correct is not None and type(correct) is not bool:
            raise ValueError("correct must be a boolean or null")
        if error and correct is not None:
            raise ValueError("A grading error must have correct=null")
        for key in ("prompt", "answer", "response"):
            if not isinstance(row[key], str):
                raise ValueError(f"{key} must be a string")
        tokens = row.get("token_count")
        if tokens is not None and (type(tokens) is not int or tokens < 0):
            raise ValueError("token_count must be nonnegative or null")
        line = json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        self.stream.write(line)
        self.digest.update(line.encode("utf-8"))
        question[sample] = (correct, row.get("predicted_answer"), row.get("formatted"))
        counts = self.counts[task]
        counts["completed_samples"] += 1
        counts["correct_samples"] += correct is True
        counts["grading_errors"] += correct is None
        counts["truncated_samples"] += row.get("finish_reason") == "length"
        counts["known_finish_reasons"] += row.get("finish_reason") is not None
        if tokens is not None:
            counts["known_token_counts"] += 1
            counts["generated_tokens"] += tokens
        if "formatted" in row:
            counts["known_format_counts"] += 1
            counts["formatted_samples"] += bool(row["formatted"])

    def scores(self, error=None):
        benchmarks = {}
        for task, (q, n) in self.specs.items():
            c, questions = self.counts[task], self.questions[task]
            received = c["completed_samples"]
            complete_questions = [
                samples
                for samples in questions.values()
                if len(samples) == n
                and all(grade is not None for grade, _, _ in samples.values())
            ]
            complete = len(complete_questions) == q
            passed = sum(
                any(grade for grade, _, _ in samples.values())
                for samples in complete_questions
            )
            metric = dict(
                complete=complete,
                expected_questions=q,
                samples_per_question=n,
                expected_samples=q * n,
                completed_samples=received,
                graded_samples=received - c["grading_errors"],
                correct_samples=c["correct_samples"],
                grading_errors=c["grading_errors"],
                complete_questions=len(complete_questions),
                passed_questions=passed,
                avg_at_n_pct=100 * c["correct_samples"] / (q * n) if complete else None,
                pass_at_n_pct=100 * passed / q if complete else None,
                truncated_samples=c["truncated_samples"]
                if received and c["known_finish_reasons"] == received
                else None,
                truncation_pct=100 * c["truncated_samples"] / received
                if received and c["known_finish_reasons"] == received
                else None,
                avg_generated_tokens=c["generated_tokens"] / received
                if received and c["known_token_counts"] == received
                else None,
            )
            if received and c["known_format_counts"] == received:
                metric.update(
                    formatted_samples=c["formatted_samples"],
                    format_rate=100 * c["formatted_samples"] / received,
                )
                # Same exact-string plurality and sample-order tie break as vOPD.
                votes = 0
                for samples in complete_questions:
                    predictions = [
                        (samples[s][1], samples[s][0])
                        for s in sorted(samples)
                        if samples[s][2]
                    ]
                    if predictions:
                        winner = Counter(p for p, _ in predictions).most_common(1)[0][0]
                        votes += next(grade for p, grade in predictions if p == winner)
                metric["majority_vote_at_n_pct"] = 100 * votes / q if complete else None
            benchmarks[task] = metric
        result = dict(self.metadata)
        result.update(
            schema_version=1,
            status="complete"
            if all(m["complete"] for m in benchmarks.values()) and not error
            else "partial",
            results_file="results.jsonl",
            results_sha256=self.digest.hexdigest(),
            benchmarks=benchmarks,
        )
        if error:
            result["error"] = str(error)
        return result

    def flush(self, error=None):
        self.stream.flush()
        result = self.scores(error)
        path = self.directory / "scores.json"
        temporary = path.with_suffix(".json.tmp")
        try:
            temporary.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return result

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.flush(f"{exc_type.__name__}: {exc_value}" if exc_type else None)
        finally:
            self.stream.close()
