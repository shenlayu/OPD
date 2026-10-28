"""Independent GPU evaluation workers with one validated aggregate result stream.

Workers get CUDA_VISIBLE_DEVICES before Python imports the model runtime. Each
worker keeps a durable JSONL stream; the parent alone owns final scores/results.
"""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from omegaconf import OmegaConf

from scripts._results import ResultWriter


def validate_parallel_config(config, module=None):
    e = config.evaluation
    size = e.get("data_parallel_size", 1)
    if type(size) is not int or size < 2:
        raise ValueError("Parallel evaluation requires data_parallel_size >= 2")
    if e.seed_mode != "per_sample" and not (module == "vOPD" and e.seed_mode == "engine"):
        raise ValueError("Parallel evaluation requires seed_mode=per_sample")
    if e.tensor_parallel_size != 1:
        raise ValueError("Data-parallel evaluation currently requires tensor_parallel_size=1")
    devices = [str(gpu) for gpu in config.run.gpus]
    if len(devices) < size or len(set(devices)) != len(devices):
        raise ValueError("Parallel evaluation requires enough distinct run.gpus")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        allowed = {device.strip() for device in visible.split(",")}
        if not set(devices[:size]).issubset(allowed):
            raise ValueError("Requested run.gpus are outside CUDA_VISIBLE_DEVICES")
    return devices[:size]


def drain_worker_rows(stream, writer):
    """Merge complete records, leaving an in-progress final line for later."""
    count = 0
    while True:
        position = stream.tell()
        line = stream.readline()
        if not line:
            break
        newline = b"\n" if isinstance(line, bytes) else "\n"
        if not line.endswith(newline):
            stream.seek(position)
            break
        writer.write(json.loads(line))
        count += 1
    return count


def _terminate_workers(processes):
    # Each worker owns a separate session, including its vLLM engine descendants.
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10
    for process in processes:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _interrupted(signum, _frame):
    raise InterruptedError(f"Evaluation interrupted by signal {signum}")


def run_parallel(config_path, config, module, checkpoint, output, tasks, metadata, initial_rows=()):
    devices = validate_parallel_config(config, module)
    if config.evaluation.seed_mode == "engine":
        metadata = dict(metadata, evaluation_execution=dict(
            mode="independent_engines_question_shards", gpus=devices,
            engine_seed=config.evaluation.seed,
            random_stream_note="Each GPU uses the configured engine seed. Question sharding changes batching and RNG streams relative to a single GPU; bitwise-identical samples are not claimed.",
        ))
    specs = {task.name: (len(rows), task.samples) for task, rows in tasks}
    shard_dir = Path(output) / "parallel_workers" / uuid.uuid4().hex
    shard_dir.mkdir(parents=True)
    processes, streams, logs = [], [], []
    previous = signal.signal(signal.SIGTERM, _interrupted)
    try:
        with ResultWriter(output, metadata, specs) as writer:
            for row in initial_rows:
                writer.write(row)
            writer.flush()
            try:
                for rank, device in enumerate(devices):
                    rows_path = shard_dir / f"worker_{rank}.jsonl"
                    rows_path.touch(exist_ok=False)
                    # Read bytes so a concurrent write cannot expose a partial
                    # UTF-8 character to the decoder before the newline arrives.
                    streams.append(rows_path.open("rb"))
                    log = (shard_dir / f"worker_{rank}.log").open("x", encoding="utf-8")
                    logs.append(log)
                    env = os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES=device, PYTHONUNBUFFERED="1")
                    process = subprocess.Popen(
                        [sys.executable, "-m", "scripts._parallel_evaluate",
                         str(config_path), module, str(checkpoint), str(rows_path),
                         str(rank), str(len(devices))],
                        env=env, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    processes.append(process)
                    print(
                        f"EVAL worker={rank} gpu={device} pid={process.pid} "
                        f"log={log.name}", flush=True,
                    )
                totals = [0] * len(devices)
                last_report = time.monotonic()
                while True:
                    changed = False
                    for rank, stream in enumerate(streams):
                        count = drain_worker_rows(stream, writer)
                        totals[rank] += count
                        changed = changed or bool(count)
                    if changed:
                        writer.flush()
                    statuses = [process.poll() for process in processes]
                    failed = [(rank, code) for rank, code in enumerate(statuses)
                              if code not in (None, 0)]
                    if failed:
                        raise RuntimeError(f"Evaluation workers failed: {failed}; logs: {shard_dir}")
                    if changed or time.monotonic() - last_report >= 30:
                        print(f"EVAL workers completed_samples={totals} total={sum(totals)}", flush=True)
                        last_report = time.monotonic()
                    if all(code == 0 for code in statuses):
                        # Workers can flush their last records between the read and poll.
                        for stream in streams:
                            drain_worker_rows(stream, writer)
                            if stream.read():
                                raise RuntimeError("Evaluation worker left an incomplete JSONL record")
                        break
                    time.sleep(0.5)
                scores = writer.flush()
                for task, _rows in tasks:
                    metric = scores["benchmarks"][task.name]
                    if not metric["complete"]:
                        raise RuntimeError(f"Evaluation incomplete or has grading errors: {task.name}")
                    print(
                        f"EVAL {task.name}: questions={metric['expected_questions']} "
                        f"samples={metric['completed_samples']} "
                        f"avg@{task.samples}={metric['avg_at_n_pct']:.4f}% "
                        f"pass@{task.samples}={metric['pass_at_n_pct']:.4f}% "
                        f"truncated={metric['truncated_samples']} "
                        f"grading_errors={metric['grading_errors']}", flush=True,
                    )
            finally:
                _terminate_workers(processes)
                # Retain every complete record produced before a failure or interrupt.
                for stream in streams:
                    drain_worker_rows(stream, writer)
    finally:
        signal.signal(signal.SIGTERM, previous)
        for stream in streams:
            stream.close()
        for log in logs:
            log.close()


class _WorkerWriter:
    def __init__(self, path):
        self.stream = Path(path).open("a", encoding="utf-8")

    def write(self, row):
        self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def flush(self):
        self.stream.flush()


def worker_main(config_path, module, checkpoint, rows_path, rank, size):
    from scripts._evaluate import load_tasks, run_evaluation

    signal.signal(signal.SIGTERM, _interrupted)
    config = OmegaConf.load(config_path)
    writer = _WorkerWriter(rows_path)
    try:
        run_evaluation(
            config, module, checkpoint, writer, load_tasks(config),
            shard_rank=int(rank), shard_world_size=int(size),
        )
    finally:
        writer.stream.close()


if __name__ == "__main__":
    worker_main(*sys.argv[1:])
