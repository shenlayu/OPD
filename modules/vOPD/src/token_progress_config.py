"""Portable configuration for Scheme B (no model/runtime imports)."""

from dataclasses import asdict, dataclass, field
import math


CATEGORIES = ("number", "entity_like", "code_like", "punctuation", "reflection", "stop", "other")


@dataclass
class ProgressConfig:
    enabled: bool = False
    entropy_edges: list = field(default_factory=lambda: [1.0, 2.0, 4.0])  # nats
    low_entropy_max_bin: int = 1
    fast_half_life: float = 50.0
    slow_half_life: float = 500.0
    min_history: int = 1000  # sufficiently populated optimizer updates per bucket
    evaluate_every: int = 10
    plateau_patience: int = 3
    epsilon: float = 0.02
    epsilon_up: float = 0.05
    plateau_beta_scale: float = 0.3
    plateau_weight: float = 0.5
    min_tokens_per_step: int = 32
    class_overrides: dict = field(default_factory=dict)  # token ID -> category

    @classmethod
    def parse(cls, values=None):
        config = cls(**dict(values or {}))
        config.entropy_edges = list(config.entropy_edges)
        config.class_overrides = {str(k): v for k, v in dict(config.class_overrides).items()}
        if type(config.enabled) is not bool:
            raise ValueError("token_gate_progress.enabled must be boolean")
        edges = config.entropy_edges
        if len(edges) not in (2, 3) or any(not math.isfinite(x) or x <= 0 for x in edges) or edges != sorted(set(edges)):
            raise ValueError("Scheme B requires 3 or 4 increasing teacher entropy bands (nats)")
        if type(config.low_entropy_max_bin) is not int or not 0 <= config.low_entropy_max_bin < len(edges):
            raise ValueError("low_entropy_max_bin must leave at least one high-entropy band")
        if not (math.isfinite(config.fast_half_life) and math.isfinite(config.slow_half_life)
                and 0 < config.fast_half_life < config.slow_half_life):
            raise ValueError("EMA half lives must satisfy 0 < fast < slow")
        for name in ("min_history", "evaluate_every", "plateau_patience", "min_tokens_per_step"):
            if type(getattr(config, name)) is not int or getattr(config, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= config.epsilon < config.epsilon_up < 1:
            raise ValueError("Progress hysteresis requires 0 <= epsilon < epsilon_up < 1")
        for name in ("plateau_beta_scale", "plateau_weight"):
            if not 0 <= getattr(config, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if any(not k.isdigit() or v not in CATEGORIES for k, v in config.class_overrides.items()):
            raise ValueError("class_overrides maps nonnegative token IDs to known categories")
        return config

    def to_dict(self):
        return asdict(self)
