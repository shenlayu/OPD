from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from typing import Any


def adapter_initialization_fingerprint(model) -> str | None:
    """Fingerprint LoRA parameters before optimization for paired-run checks."""
    import torch

    digest = hashlib.sha256()
    found = False
    for name, parameter in sorted(model.named_parameters()):
        if "lora_" not in name:
            continue
        found = True
        tensor = parameter.detach().cpu().contiguous()
        digest.update(f"{name}\0{tuple(tensor.shape)}\0{tensor.dtype}\0".encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest() if found else None


CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}{% endif %}"
)


def safe_init_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(cls.__init__)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    return {k: v for k, v in kwargs.items() if k in allowed and v is not None}


def resolve_dtype(dtype: str) -> Any:
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = dtype.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported --torch_dtype: {dtype}")
    return mapping[key]


def normalize_dtype_str(dtype: str) -> str:
    key = dtype.lower()
    aliases = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }
    return aliases.get(key, key)


def extract_problem(example: dict[str, Any], dataset_name: str | None = None) -> str:
    candidate_keys = ("problem", "Question", "question", "prompt")
    for key in dict.fromkeys(candidate_keys):
        value = example.get(key, None)
        if isinstance(value, str) and value.strip():
            return value
    raise KeyError(
        f"Could not find problem text in sample keys: {list(example.keys())}"
    )


def format_prompt(problem: str) -> str:
    return f"{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def ensure_chat_template(tokenizer: Any) -> bool:
    if getattr(tokenizer, "chat_template", None):
        return False
    tokenizer.chat_template = CHAT_TEMPLATE
    return True


def format_prompt_with_chat_template(
    tokenizer: Any, problem: str, enable_thinking: bool = False
) -> str:
    ensure_chat_template(tokenizer)
    user_message = format_prompt(problem)
    messages = [{"role": "user", "content": user_message}]
    try:
        # Qwen-style tokenizers may support `enable_thinking`.
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def format_prompt_for_model(
    tokenizer: Any,
    problem: str,
    use_chat_template: bool | None = None,
    enable_thinking: bool = False,
) -> str:
    if use_chat_template:
        return format_prompt_with_chat_template(
            tokenizer, problem, enable_thinking=enable_thinking
        )
    return format_prompt(problem)


@dataclass
class DistillModeConfig:
    rkl_advantage: bool
    single_step_decomposition: bool
    gamma: Any


def distill_mode_to_config(mode: str) -> DistillModeConfig:
    mode = mode.lower()
    if mode == "tml":
        # Think Machine Lab style on-policy KD per TRL MiniLLM docs.
        return DistillModeConfig(
            rkl_advantage=True, single_step_decomposition=False, gamma=False
        )
    if mode == "gkd_rkl":
        # Reverse-KL single-step distillation signal.
        return DistillModeConfig(
            rkl_advantage=False, single_step_decomposition=True, gamma=None
        )
    raise ValueError(f"Unknown distill mode: {mode}")


def effective_distill_mode(args: Any) -> str:
    if args.distill_mode is not None:
        return args.distill_mode
    if args.use_baseline:
        return "tml"
    raise AssertionError("--distill_mode must be set when --use_baseline is false")


def load_training_args(argv):
    from omegaconf import OmegaConf

    if len(argv) != 1:
        raise ValueError(
            "Use scripts/run_vopd.py; the worker expects one resolved config"
        )
    return OmegaConf.load(argv[0]).training


def load_training_dataset_by_alias(dataset_name, split="train"):
    from scripts._data import load_rows

    return load_rows(dataset_name, split)
