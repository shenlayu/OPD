"""Final-checkpoint evaluation with one scores.json and one results.jsonl."""

import signal
import sys
from pathlib import Path

from omegaconf import OmegaConf

from scripts._data import boxed_answer, load_rows, problem_and_answer
from scripts._results import ResultWriter


def stop_token_ids(tokenizer):
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    for token in ["<|im_end|>", "<|endoftext|>"]:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) == 1 and encoded[0] in tokenizer.all_special_ids:
            ids.add(encoded[0])
    return sorted(ids)


def format_prompt(tokenizer, problem, evaluation):
    style = evaluation.prompt_style
    if style in {"plain_math", "chat_math"}:
        problem += "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    if style == "plain_math":
        return problem
    if style not in {"chat_math", "chat_author"}:
        raise ValueError(f"Unknown evaluation prompt_style: {style}")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": problem}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=evaluation.enable_thinking,
    )


def grade_vopd(response, answer):
    from math_verify import parse, verify

    prediction = boxed_answer(response)
    if prediction is None:
        return False
    answer = boxed_answer(answer) or answer
    try:
        pred = prediction if "$" in prediction else f"${prediction}$"
        gold = answer if "$" in answer else f"${answer}$"
        return bool(
            verify(
                parse(gold, fallback_mode="no_fallback"),
                parse(pred, fallback_mode="no_fallback"),
                timeout_seconds=5,
            )
        )
    except Exception:
        normalize = lambda x: x.replace("$", "").replace(" ", "").lower().strip()
        return normalize(prediction) == normalize(answer)


def grade(response, answer, module, timeout):
    def timed_out(*_):
        raise TimeoutError("Answer grading timed out")

    previous = signal.signal(signal.SIGALRM, timed_out)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        if module == "vOPD":
            return grade_vopd(response, answer)
        from modules.rethinking.grading import grade_answer_verl

        return bool(grade_answer_verl(response, answer))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def evaluate_tasks(
    config, module, tokenizer, llm, sampling_params, writer, tasks, lora_request=None,
    *, shard_rank=0, shard_world_size=1, skip_samples=frozenset(),
):
    e = config.evaluation
    if not 0 <= shard_rank < shard_world_size:
        raise ValueError("Invalid evaluation shard rank or world size")
    if shard_world_size > 1 and e.seed_mode not in {"per_sample", "engine"}:
        raise ValueError("Unknown parallel evaluation seed mode")
    for task_index, (task, rows) in enumerate(tasks):
        requests = []
        completed = 0
        if (
            e.seed_mode == "per_sample"
            and len(rows) * e.seed_example_stride > e.seed_task_stride
        ):
            raise ValueError(
                "Evaluation seed ranges overlap; increase seed_task_stride"
            )

        def flush():
            nonlocal completed
            if not requests:
                return
            params = [
                sampling_params(
                    temperature=e.temperature,
                    top_p=e.top_p,
                    top_k=e.top_k,
                    min_p=e.min_p,
                    presence_penalty=e.presence_penalty,
                    repetition_penalty=e.repetition_penalty,
                    max_tokens=e.max_new_tokens,
                    stop_token_ids=stop_token_ids(tokenizer)
                    if e.stop_mode == "registered_special"
                    else None,
                    seed=request["seed"],
                    n=task.samples if e.seed_mode == "engine" else 1,
                )
                for request in requests
            ]
            outputs = llm.generate(
                [r["formatted_prompt"] for r in requests],
                params,
                lora_request=lora_request,
                use_tqdm=False,
            )
            if len(outputs) > len(requests):
                raise RuntimeError("Generation returned unexpected requests")
            for request, output in zip(requests, outputs):
                expected = task.samples if e.seed_mode == "engine" else 1
                if len(output.outputs) > expected:
                    raise RuntimeError("Generation returned unexpected samples")
                for offset, completion in enumerate(output.outputs):
                    row = {k: v for k, v in request.items() if k != "formatted_prompt"}
                    row["sample_id"] = (
                        offset if e.seed_mode == "engine" else request["sample_id"]
                    )
                    if (task.name, row["example_id"], row["sample_id"]) in skip_samples:
                        continue
                    prediction = boxed_answer(completion.text)
                    row.update(
                        response=completion.text,
                        predicted_answer=prediction,
                        token_count=len(completion.token_ids),
                        finish_reason=completion.finish_reason,
                        stop_reason=completion.stop_reason,
                        correct=None,
                        grading_error=None,
                    )
                    if module == "vOPD":
                        row["formatted"] = prediction is not None
                        row["predicted_answer"] = (
                            prediction or "[No boxed answer found]"
                        )
                    try:
                        row["correct"] = grade(
                            completion.text, row["answer"], module, e.grading_timeout
                        )
                    except Exception as error:
                        row["grading_error"] = f"{type(error).__name__}: {error}"
                        print(
                            f"Grading error: {task.name}[{row['example_id']},{row['sample_id']}]: {error}",
                            flush=True,
                        )
                    writer.write(row)
                if len(output.outputs) != expected:
                    raise RuntimeError("Generation returned an incomplete sample group")
            writer.flush()
            if len(outputs) != len(requests):
                raise RuntimeError("Generation returned an incomplete batch")
            completed += len(requests)
            if shard_world_size > 1:
                print(
                    f"EVAL worker={shard_rank} {task.name}: "
                    f"completed_samples={completed}", flush=True,
                )
            requests.clear()

        for index, row in enumerate(rows):
            problem, answer = problem_and_answer(row)
            prompt = format_prompt(tokenizer, problem, e)
            if (
                len(tokenizer.encode(prompt, add_special_tokens=False))
                > e.max_prompt_length
            ):
                raise ValueError(
                    f"Prompt exceeds evaluation budget: {task.name}[{index}]"
                )
            samples = range(task.samples) if e.seed_mode == "per_sample" else [0]
            for sample in samples:
                if e.seed_mode == "engine":
                    if all((task.name, index, s) in skip_samples for s in range(task.samples)):
                        continue
                    shard_key = index
                else:
                    if (task.name, index, sample) in skip_samples:
                        continue
                    shard_key = index * task.samples + sample
                if shard_key % shard_world_size != shard_rank:
                    continue
                seed = (
                    e.seed
                    + task_index * e.seed_task_stride
                    + index * e.seed_example_stride
                    + sample
                    if e.seed_mode == "per_sample"
                    else None
                )
                requests.append(
                    dict(
                        benchmark=task.name,
                        example_id=index,
                        sample_id=sample,
                        prompt=problem,
                        formatted_prompt=prompt,
                        answer=answer,
                        seed=seed,
                    )
                )
                if len(requests) == e.batch_size:
                    flush()
        flush()
        if shard_world_size > 1:
            continue  # The parent validates full benchmark coverage after merging.
        metric = writer.scores()["benchmarks"][task.name]
        if not metric["complete"]:
            raise RuntimeError(
                f"Evaluation incomplete or has grading errors: {task.name}"
            )
        print(
            f"EVAL {task.name}: questions={len(rows)} samples={metric['completed_samples']} "
            f"avg@{task.samples}={metric['avg_at_n_pct']:.4f}% "
            f"pass@{task.samples}={metric['pass_at_n_pct']:.4f}% "
            f"truncated={metric['truncated_samples']} grading_errors={metric['grading_errors']}",
            flush=True,
        )


def load_tasks(config):
    e = config.evaluation
    tasks = []
    for task in e.datasets:
        rows = load_rows(task.path, task.split)
        if task.limit is not None:
            rows = rows.select(range(min(task.limit, len(rows))))
        if not len(rows):
            raise ValueError(f"Empty evaluation dataset: {task.name}")
        tasks.append((task, rows))
    return tasks


def run_evaluation(
    config, module, checkpoint, writer, tasks, *, shard_rank=0, shard_world_size=1,
    skip_samples=frozenset(),
):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    e = config.evaluation
    adapter = module == "vOPD" and config.training.use_peft
    base = config.training.model_name_or_path if adapter else checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    kwargs = dict(
        model=base,
        tokenizer=checkpoint,
        dtype=e.dtype,
        trust_remote_code=True,
        tensor_parallel_size=e.tensor_parallel_size,
        gpu_memory_utilization=e.gpu_memory_utilization,
        max_model_len=e.max_model_len,
        max_num_seqs=e.batch_size,
        max_num_batched_tokens=e.max_num_batched_tokens,
        enforce_eager=e.enforce_eager,
        seed=e.seed,
    )
    lora_request = None
    if adapter:
        from vllm.lora.request import LoRARequest

        if not (Path(checkpoint) / "adapter_model.safetensors").exists():
            raise FileNotFoundError("Final LoRA checkpoint has no adapter weights")
        kwargs.update(
            enable_lora=True, max_lora_rank=config.training.lora_r, max_loras=1
        )
        lora_request = LoRARequest("final", 1, checkpoint)
    llm = LLM(**kwargs)
    try:
        evaluate_tasks(
            config, module, tokenizer, llm, SamplingParams, writer, tasks,
            lora_request, shard_rank=shard_rank, shard_world_size=shard_world_size,
            skip_samples=skip_samples,
        )
    finally:
        llm.llm_engine.engine_core.shutdown()


def main(config_path, module, checkpoint, output=None):
    config = OmegaConf.load(config_path)
    e = config.evaluation
    tasks = load_tasks(config)
    metadata = dict(
        module=module,
        dataset=config.run.dataset,
        checkpoint=str(checkpoint),
        evaluation=OmegaConf.to_container(e, resolve=True),
        config_file=f"configs/{Path(config_path).name}",
    )
    output = output or Path(config_path).resolve().parent.parent
    if e.get("data_parallel_size", 1) > 1:
        from scripts._parallel_evaluate import run_parallel

        return run_parallel(config_path, config, module, checkpoint, output, tasks, metadata)
    with ResultWriter(
        output, metadata, {t.name: (len(rows), t.samples) for t, rows in tasks}
    ) as writer:
        run_evaluation(config, module, checkpoint, writer, tasks)


if __name__ == "__main__":
    main(*sys.argv[1:])
