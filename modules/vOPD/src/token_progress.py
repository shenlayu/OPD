"""Teacher-only buckets and optimizer-step learning-progress memory (Scheme B).

The monitored discrepancy is full-vocabulary, symmetric JS in nats, before
clipping or loss reweighting. Statistics never participate in autograd.
"""

import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

import torch

from .token_progress_config import CATEGORIES, ProgressConfig


STATE_FILE = "token_progress_state.json"
PROGRESS_STATS = ("progress_bucket", "progress_js", "progress_plateau", "progress_weight")
REFLECTION_WORDS = {"wait", "however", "alternatively", "rethink", "check", "verify"}


def lexical_category(text):
    """Reproducible token-piece proxies, not an entity recognizer or code parser."""
    text = text.strip()
    if not text:
        return "other"
    if text.casefold() in REFLECTION_WORDS:
        return "reflection"
    if any(c.isnumeric() for c in text) and not any(c.isalpha() for c in text):
        return "number"
    if all(unicodedata.category(c)[0] in "PS" for c in text):
        return "punctuation"
    if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", text) and (
        "_" in text or any(c.isdigit() for c in text) or re.search(r"[a-z][A-Z]", text)
    ):
        return "code_like"
    if text.isalpha() and text[0].isupper():
        return "entity_like"
    return "other"


def build_token_classes(tokenizer, vocab_size, stop_ids, overrides):
    classes = [CATEGORIES.index("other")] * vocab_size
    vocabulary = sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])
    for piece, token_id in vocabulary:
        if not 0 <= token_id < vocab_size:
            raise ValueError("Tokenizer ID is outside model vocabulary")
        classes[token_id] = CATEGORIES.index(lexical_category(tokenizer.convert_tokens_to_string([piece])))
    for key, category in overrides.items():
        if int(key) >= vocab_size:
            raise ValueError("class_overrides ID is outside model vocabulary")
        classes[int(key)] = CATEGORIES.index(category)
    # Explicit stop/control IDs always retain protection, even with overrides.
    for token_id in stop_ids:
        if not 0 <= token_id < vocab_size:
            raise ValueError("Stop ID is outside model vocabulary")
        classes[token_id] = CATEGORIES.index("stop")
    fingerprint = hashlib.sha256(json.dumps([vocabulary, classes], ensure_ascii=True).encode()).hexdigest()
    return classes, fingerprint


class TokenProgress:
    def __init__(self, config, token_classes, *, classifier_fingerprint=None, context=None):
        self.config = ProgressConfig.parse(config)
        if not self.config.enabled:
            raise ValueError("Instantiate TokenProgress only when Scheme B is enabled")
        self.token_classes = torch.tensor(token_classes, dtype=torch.long)
        if not token_classes or any(type(c) is not int or not 0 <= c < len(CATEGORIES) for c in token_classes):
            raise ValueError("Invalid token class map")
        self.classifier_fingerprint = classifier_fingerprint or hashlib.sha256(bytes(token_classes)).hexdigest()
        self.context = dict(context or {})
        self.bands = len(self.config.entropy_edges) + 1
        self.size = len(CATEGORIES) * self.bands
        self.fast = torch.zeros(self.size, dtype=torch.float64)
        self.slow = self.fast.clone()
        self.observed_steps = torch.zeros(self.size, dtype=torch.long)
        self.token_counts = self.observed_steps.clone()
        self.low_streak = self.observed_steps.clone()
        self.since_evaluation = self.observed_steps.clone()
        self.plateau = torch.zeros(self.size, dtype=torch.bool)
        self.global_step = 0
        self.pending = None
        self._cache = {}

    def _device_state(self, device):
        key = str(device)
        if key not in self._cache:
            self._cache[key] = (
                self.token_classes.to(device),
                torch.tensor(self.config.entropy_edges, device=device),
                self.plateau.to(device).clone(),
            )
        return self._cache[key]

    @torch.no_grad()
    def policy(self, lp, lq, best, beta):
        """Pure policy during checkpoint recomputation; observation happens outside."""
        classes, edges, plateaus = self._device_state(lq.device)
        entropy = -(lq.exp() * lq).sum(-1)
        band = torch.bucketize(entropy, edges.to(entropy.dtype), right=True)
        category = classes[best]
        bucket = category * self.bands + band
        protected = (category == CATEGORIES.index("stop")) | (category == CATEGORIES.index("reflection"))
        active = plateaus[bucket] & ~protected
        low = band <= self.config.low_entropy_max_bin
        adjusted = torch.where(active & low, beta * self.config.plateau_beta_scale, beta)
        adjusted = torch.where(active & ~low, torch.zeros_like(beta), adjusted)
        weight = torch.where(active & low, beta.new_tensor(self.config.plateau_weight), torch.ones_like(beta))
        lm = torch.logaddexp(lp, lq) - math.log(2)
        discrepancy = (0.5 * (lp.exp() * (lp - lm) + lq.exp() * (lq - lm)).sum(-1)).clamp_min(0)
        observations = torch.stack((bucket.to(beta.dtype), discrepancy, active.to(beta.dtype), weight), -1)
        return adjusted, weight, observations

    @torch.no_grad()
    def observe(self, observations):
        """Accumulate each valid token once, outside autograd checkpoint closures."""
        if self.pending is None:
            self.pending = torch.zeros((2, self.size), dtype=torch.float64, device=observations.device)
        ids = observations[:, 0].long()
        self.pending[0].scatter_add_(0, ids, observations[:, 1].double())
        self.pending[1].scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.float64))

    @torch.no_grad()
    def commit(self, global_step, device):
        """One collective per optimizer update, including ranks with zero tokens."""
        if global_step != self.global_step + 1:
            raise ValueError(f"Scheme B expected step {self.global_step + 1}, got {global_step}")
        totals = self.pending if self.pending is not None else torch.zeros((2, self.size), dtype=torch.float64, device=device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(totals)
        totals = totals.cpu()
        if not torch.isfinite(totals).all():
            raise ValueError("Non-finite Scheme B discrepancy statistics")
        self.pending = None
        counts = totals[1].long()
        valid = counts >= self.config.min_tokens_per_step
        means = totals[0] / counts.clamp_min(1)
        first = valid & (self.observed_steps == 0)
        for estimate, half_life in ((self.fast, self.config.fast_half_life), (self.slow, self.config.slow_half_life)):
            decay = 2 ** (-1 / half_life)
            estimate[valid] = decay * estimate[valid] + (1 - decay) * means[valid]
            estimate[first] = means[first]
        self.token_counts += counts
        self.observed_steps += valid.long()
        self.since_evaluation += valid.long()
        self.global_step = global_step
        evaluate = global_step % self.config.evaluate_every == 0
        if evaluate:
            eligible = (self.observed_steps >= self.config.min_history) & (self.since_evaluation > 0)
            progress = self.progress()
            low = eligible & (progress < self.config.epsilon)
            self.low_streak = torch.where(low, self.low_streak + 1, torch.zeros_like(self.low_streak))
            self.plateau |= low & (self.low_streak >= self.config.plateau_patience)
            self.plateau[eligible & (progress > self.config.epsilon_up)] = False
            self.since_evaluation.zero_()
        self._cache.clear()
        return self.report() if evaluate else None

    def progress(self):
        return 1 - self.fast / self.slow.clamp_min(1e-12)

    def report(self):
        progress = self.progress()
        return dict(global_step=self.global_step, buckets=[dict(
            category=CATEGORIES[b // self.bands], entropy_bin=b % self.bands,
            tokens=int(self.token_counts[b]), observed_steps=int(self.observed_steps[b]),
            fast=float(self.fast[b]), slow=float(self.slow[b]), progress=float(progress[b]),
            low_streak=int(self.low_streak[b]), plateau=bool(self.plateau[b]),
        ) for b in range(self.size)])

    def state_dict(self, clip):
        if self.pending is not None:
            raise ValueError("Save Scheme B state only at an optimizer-step boundary")
        return dict(version=1, config=self.config.to_dict(), context=self.context,
                    classifier_fingerprint=self.classifier_fingerprint, global_step=self.global_step,
                    clip=clip, **{name: getattr(self, name).tolist() for name in (
                        "fast", "slow", "observed_steps", "token_counts", "low_streak", "since_evaluation", "plateau")})

    def save(self, directory, clip):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / STATE_FILE
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.state_dict(clip), indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    def load(self, directory, clip=None):
        data = json.loads((Path(directory) / STATE_FILE).read_text())
        expected = dict(version=1, config=self.config.to_dict(), context=self.context,
                        classifier_fingerprint=self.classifier_fingerprint)
        if any(data.get(k) != v for k, v in expected.items()):
            raise ValueError("Scheme B checkpoint configuration, objective or tokenizer mismatch")
        saved_clip = data.get("clip")
        if not isinstance(saved_clip, (int, float)) or not math.isfinite(saved_clip) or saved_clip <= 0:
            raise ValueError("Scheme B checkpoint has no valid frozen clip threshold")
        if clip is not None and clip != saved_clip:
            raise ValueError("Scheme B checkpoint clip threshold mismatch")
        step = data.get("global_step")
        if type(step) is not int or step < 0:
            raise ValueError("Invalid Scheme B checkpoint step")
        restored = {}
        for name in ("fast", "slow", "observed_steps", "token_counts", "low_streak", "since_evaluation", "plateau"):
            values = data.get(name)
            if not isinstance(values, list) or len(values) != self.size:
                raise ValueError(f"Invalid Scheme B checkpoint field: {name}")
            dtype = getattr(self, name).dtype
            if dtype == torch.bool and any(type(v) is not bool for v in values):
                raise ValueError(f"Invalid Scheme B checkpoint field: {name}")
            if dtype == torch.long and any(type(v) is not int for v in values):
                raise ValueError(f"Invalid Scheme B checkpoint field: {name}")
            value = torch.tensor(values, dtype=dtype)
            if not torch.isfinite(value).all() or (value < 0).any():
                raise ValueError(f"Invalid Scheme B checkpoint field: {name}")
            restored[name] = value
        if (restored["observed_steps"] > step).any():
            raise ValueError("Scheme B history exceeds checkpoint step")
        for name, value in restored.items():
            setattr(self, name, value)
        self.global_step, self.pending = step, None
        self._cache.clear()
        return saved_clip
