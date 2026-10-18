import os
import sys
from typing import Any

from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, set_seed

from trl.experimental.minillm import MiniLLMConfig
from modules.vOPD.src.opd_trainer import CustomMiniLLMTrainer as MiniLLMTrainer
from modules.vOPD.src.util import (
    adapter_initialization_fingerprint,
    distill_mode_to_config,
    effective_distill_mode as get_effective_distill_mode,
    extract_problem,
    format_prompt_for_model,
    load_training_args,
    load_training_dataset_by_alias,
    normalize_dtype_str,
    resolve_dtype,
    safe_init_kwargs,
)


if __name__ == "__main__":
    args = load_training_args(sys.argv[1:])
    # GRPO creates PEFT before Transformers Trainer seeds itself. Seed here so
    # independently launched objectives start with identical LoRA parameters.
    set_seed(args.seed)
    if "wandb" in [target.strip() for target in args.report_to.split(",")]:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    teacher_model = args.teacher_model_name_or_path or args.model_name_or_path
    effective_distill_mode = get_effective_distill_mode(args)
    mode_cfg = distill_mode_to_config(effective_distill_mode)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    output_dir = args.output_dir
    run_name = args.run_name
    resume_from_checkpoint = args.resume_from_checkpoint
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model)
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError(
            "Dense distillation requires aligned student/teacher token IDs"
        )
    del teacher_tokenizer

    def _to_train_prompt(ex: dict[str, Any]) -> dict[str, str]:
        problem = extract_problem(ex, args.dataset_name)
        prompt = format_prompt_for_model(
            tokenizer,
            problem,
            use_chat_template=args.use_chat_template,
            enable_thinking=args.enable_thinking,
        )
        return {"prompt": prompt}

    train_dataset = load_training_dataset_by_alias(
        args.dataset_name, split=args.train_split
    )
    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(
            range(min(args.max_train_samples, len(train_dataset)))
        )
    train_dataset = train_dataset.map(
        _to_train_prompt,
        remove_columns=train_dataset.column_names,
    )

    eval_dataset = None
    if args.eval_split is not None:
        eval_dataset = load_training_dataset_by_alias(
            args.dataset_name, split=args.eval_split
        )
        if args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(
                range(min(args.max_eval_samples, len(eval_dataset)))
            )
        eval_dataset = eval_dataset.map(
            _to_train_prompt,
            remove_columns=eval_dataset.column_names,
        )

    dtype_str = normalize_dtype_str(args.torch_dtype)
    dtype_obj = resolve_dtype(args.torch_dtype)

    model_init_kwargs = {
        "attn_implementation": args.attn_implementation,
        # Keep both for compatibility across TRL/Transformers versions.
        "dtype": dtype_str,
        "torch_dtype": dtype_obj,
        "trust_remote_code": True,
        "use_cache": False if args.gradient_checkpointing else True,
    }
    teacher_model_init_kwargs = {
        "attn_implementation": args.attn_implementation,
        # MiniLLMTrainer in your installed TRL expects this exact key.
        "dtype": dtype_str,
        "torch_dtype": dtype_obj,
        "trust_remote_code": True,
        "use_cache": True,
    }

    config_kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_checkpointing_kwargs": (
            {"use_reentrant": False} if args.gradient_checkpointing else None
        ),
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "do_eval": eval_dataset is not None,
        "eval_strategy": "steps" if eval_dataset is not None else "no",
        "evaluation_strategy": "steps" if eval_dataset is not None else "no",
        "eval_steps": args.eval_steps,
        "save_total_limit": args.save_total_limit,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "kd_temperature": args.kd_temperature,
        "top_p": args.top_p,
        "disable_tqdm": True,
        # vLLM uses -1 to disable top-k, while Transformers generation uses 0.
        "top_k": args.top_k if args.use_vllm else max(0, args.top_k),
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "num_generations": args.num_generations,
        "rkl_advantage": mode_cfg.rkl_advantage,
        "single_step_decomposition": mode_cfg.single_step_decomposition,
        "gamma": args.gamma if args.gamma is not None else mode_cfg.gamma,
        "beta": args.beta,
        "use_vllm": args.use_vllm,
        "vllm_mode": args.vllm_mode,
        "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
        "vllm_max_model_length": args.max_prompt_length + args.max_completion_length,
        "bf16": dtype_str == "bfloat16",
        "ddp_find_unused_parameters": False,
        "report_to": args.report_to,
        "run_name": run_name,
        "seed": args.seed,
        "model_init_kwargs": model_init_kwargs,
        "teacher_model_init_kwargs": teacher_model_init_kwargs,
    }
    training_args = MiniLLMConfig(**safe_init_kwargs(MiniLLMConfig, config_kwargs))

    peft_config = None
    if args.use_peft:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=list(args.lora_target_modules),
            bias="none",
        )

    trainer = MiniLLMTrainer(
        model=args.model_name_or_path,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        use_baseline=args.use_baseline,
        kl_top_k=args.kl_top_k,
        opd_top_k=args.opd_top_k,
        reward_clip_lambda=args.reward_clip_lambda,
        log_grad_variance=args.log_grad_variance,
        grad_variance_logging_steps=args.grad_variance_logging_steps,
        dense_objective=getattr(args, "dense_objective", "upstream"),
        alpha_start=getattr(args, "alpha_start", 0.0),
        alpha_end=getattr(args, "alpha_end", 1.0),
        alpha_schedule=getattr(args, "alpha_schedule", "linear"),
        loss_token_chunk_size=getattr(args, "loss_token_chunk_size", 128),
        prompt_length_limit=args.max_prompt_length,
    )

    # TRL materializes the teacher dtype in-place while loading the model.
    # Restore serializable config metadata after loading, before callbacks log it.
    for init_kwargs in [
        training_args.model_init_kwargs,
        training_args.teacher_model_init_kwargs,
    ]:
        if init_kwargs:
            for key in ["dtype", "torch_dtype"]:
                if key in init_kwargs:
                    init_kwargs[key] = str(init_kwargs[key]).removeprefix("torch.")
    import torch.distributed as distributed

    initial_fingerprint = adapter_initialization_fingerprint(trainer.model)
    rank_fingerprints = [initial_fingerprint]
    if distributed.is_available() and distributed.is_initialized():
        rank_fingerprints = [None] * distributed.get_world_size()
        distributed.all_gather_object(rank_fingerprints, initial_fingerprint)
    if len(set(rank_fingerprints)) != 1:
        raise RuntimeError(
            f"LoRA initialization differs across ranks: {rank_fingerprints}"
        )
    expected_fingerprint = getattr(args, "expected_initial_adapter_sha256", None)
    if expected_fingerprint is not None and initial_fingerprint != expected_fingerprint:
        raise RuntimeError(
            f"Paired LoRA initialization mismatch: expected {expected_fingerprint}, got {initial_fingerprint}"
        )
    if args.use_peft and initial_fingerprint is None:
        raise RuntimeError(
            "PEFT enabled but no LoRA parameters were found for the initialization check"
        )
    if trainer.is_world_process_zero():
        print(
            f"Initial adapter SHA256: {initial_fingerprint}; all ranks match",
            flush=True,
        )
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_dir)
        print(
            f"Training complete: step={trainer.state.global_step}; metrics={result.metrics}",
            flush=True,
        )
