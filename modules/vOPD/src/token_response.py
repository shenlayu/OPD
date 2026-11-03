"""Detached response functions and local/trajectory gates for the q05 sweep.

The lambda=0 surrogate has the original clipped normalized-JSD gradient,
including its beta floor and exact RKL/FKL endpoints. Logged loss values are
the reference clipped JSD, not a primitive of the modified response function.
"""

from dataclasses import asdict, dataclass, fields
import math

import torch
import torch.nn.functional as F


RESPONSE_STATS = ("response_preference_delta", "response_future_scale",
                  "response_reflection", "response_stop", "response_modified_fraction")


@dataclass(frozen=True)
class ResponseConfig:
    enabled: bool = False
    power: float = 0.0
    preference: bool = False
    future_horizon: int = 0
    future_kappa: float = 1.0
    reflection_ids: tuple = ()
    stop_tau: float | None = None

    @classmethod
    def parse(cls, value=None):
        if isinstance(value, cls):
            return value
        values = dict(value or {})
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown response options: {sorted(unknown)}")
        values["reflection_ids"] = tuple(values.get("reflection_ids", ()))
        obj = cls(**values)
        if type(obj.enabled) is not bool or type(obj.preference) is not bool:
            raise ValueError("Response enabled/preference must be boolean")
        if not math.isfinite(obj.power) or not -1 <= obj.power <= 1:
            raise ValueError("Response power must be finite in [-1, 1]")
        if type(obj.future_horizon) is not int or obj.future_horizon < 0:
            raise ValueError("Response future_horizon must be a nonnegative integer")
        if not math.isfinite(obj.future_kappa) or obj.future_kappa <= 0:
            raise ValueError("Response future_kappa must be positive")
        if obj.stop_tau is not None and not 0 <= obj.stop_tau <= 1:
            raise ValueError("Response stop_tau must be in [0, 1]")
        ids = obj.reflection_ids
        if len(ids) != len(set(ids)) or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError("Response reflection IDs must be distinct nonnegative integers")
        if not obj.enabled and (obj.power or obj.preference or obj.future_horizon or ids or obj.stop_tau is not None):
            raise ValueError("Response modifications require enabled=true")
        return obj

    def to_dict(self):
        result = asdict(self)
        result["reflection_ids"] = list(self.reflection_ids)
        return result

    def validate_context(self, kind, split_stop, progress, ce_weight, stop_ids):
        if not self.enabled:
            return
        if kind != "token_jsd" or split_stop or progress or ce_weight is not None:
            raise ValueError("Response sweep requires token_jsd without split-stop, Scheme B or CE")
        if set(self.reflection_ids) & set(stop_ids):
            raise ValueError("Reflection and stop IDs must be disjoint")


@torch.no_grad()
def adjust_gate(lp, lq, best, qmax, beta, tau_q, stop_ids, config, sampled=None, future=None):
    original = beta
    reflection = torch.zeros_like(best, dtype=torch.bool)
    stopping = torch.zeros_like(reflection)
    for i in config.reflection_ids:
        reflection |= best == i
    for i in stop_ids:
        stopping |= best == i
    if config.stop_tau is not None:
        threshold = torch.where(stopping, qmax.new_tensor(config.stop_tau), qmax.new_tensor(tau_q))
        c = lp.gather(-1, best.unsqueeze(-1)).squeeze(-1).exp()
        beta = (qmax >= threshold) * (1 - c)
    beta = torch.where(reflection, torch.zeros_like(beta), beta)
    delta = torch.ones_like(beta)
    if config.preference:
        chosen = lp.argmax(-1)
        log_ratio = lq.gather(-1, chosen.unsqueeze(-1)).squeeze(-1) - lq.gather(-1, best.unsqueeze(-1)).squeeze(-1)
        delta = (-torch.expm1(log_ratio)).clamp(0, 1)
        beta = beta * delta
    scale = torch.ones_like(beta)
    if config.future_horizon:
        if sampled is None or future is None:
            raise ValueError("Future response gating requires sampled IDs and whole-sequence future statistics")
        selected = (beta > 0) & (sampled != best)
        scale = torch.where(selected, future, scale)
        beta = beta * scale
    stats = torch.stack((delta, scale, reflection.to(beta.dtype), stopping.to(beta.dtype),
                         (beta != original).to(beta.dtype)), -1)
    return beta, stats


@torch.no_grad()
def future_factors(discrepancy, mask, horizon, kappa):
    """Average available valid tokens in (t, t+H], never crossing responses.

    Empty suffixes and all-zero-discrepancy sequences retain the original gate.
    Short suffixes use their observed count; no padding is counted as evidence.
    """
    valid = mask.bool()
    d = torch.where(valid, discrepancy, torch.zeros_like(discrepancy))
    dtype = d.dtype
    sums = F.pad(d.cumsum(-1), (1, 0))
    counts = F.pad(valid.to(dtype).cumsum(-1), (1, 0))
    start = torch.arange(d.shape[-1], device=d.device) + 1
    end = (start + horizon).clamp_max(d.shape[-1])
    count = counts[..., end] - counts[..., start]
    future = (sums[..., end] - sums[..., start]).clamp_min(0) / count.clamp_min(1)
    average = d.sum(-1, keepdim=True) / valid.sum(-1, keepdim=True).clamp_min(1)
    factor = (future / average.clamp_min(torch.finfo(dtype).eps)).clamp(0, 1).pow(kappa)
    return torch.where(valid & (count > 0) & (average > torch.finfo(dtype).eps), factor, torch.ones_like(factor))


@torch.no_grad()
def sequence_future_factors(student, teacher, mask, config, chunk_size):
    """Raw symmetric JS in nats, before gating/clipping; no extra model forward."""
    dtype = torch.float64 if student.dtype == torch.float64 else torch.float32
    s = student.detach().reshape(-1, student.shape[-1])
    t = teacher.detach().reshape_as(s)
    valid = mask.reshape(-1).bool()
    values = torch.zeros(valid.shape, device=s.device, dtype=dtype)
    for offset in range(0, len(s), chunk_size):
        sl = slice(offset, offset + chunk_size)
        keep = valid[sl]
        if not keep.any():
            continue
        lp = F.log_softmax(s[sl][keep], -1, dtype=dtype)
        lq = F.log_softmax(t[sl][keep], -1, dtype=dtype)
        lm = torch.logaddexp(lp, lq) - math.log(2)
        values[sl][keep] = (.5 * (lp.exp() * (lp - lm) + lq.exp() * (lq - lm))).sum(-1).clamp_min(0)
    return future_factors(values.reshape_as(mask), mask, config.future_horizon, config.future_kappa)


def response_surrogate(lp, lq, beta, clipped, power):
    """Freeze psi and its p-mean; backpropagate only through the live p."""
    with torch.no_grad():
        b = beta.clamp(1e-3, 1 - 1e-3).unsqueeze(-1)
        log_r = lq - lp.detach()
        log_z = torch.logaddexp(b.log(), torch.log1p(-b) + log_r)
        psi = log_z if power == 0 else torch.expm1(power * log_z) / power
        # Preserve the original T2 response in the original clipped region.
        # Subtract the vocabulary-constant b before centering to reduce FP32
        # cancellation near beta=1. This leaves the gradient unchanged.
        clipped_psi = log_z + b * torch.expm1(-log_z)
        psi = torch.where(clipped.unsqueeze(-1), clipped_psi, psi) / (1 - b)
        psi = torch.where((beta == 0).unsqueeze(-1), log_r, psi)
        # Evaluate exact FKL directly below, avoiding q/p overflow at beta=1.
        psi = torch.where((beta == 1).unsqueeze(-1), torch.zeros_like(psi), psi)
        centered = psi - (lp.detach().exp() * psi).sum(-1, keepdim=True)
    surrogate = -(lp.exp() * centered).sum(-1)
    fkl = (lq.exp() * (lq - lp)).sum(-1)
    endpoint = torch.where(clipped, torch.zeros_like(fkl), fkl)
    return torch.where(beta == 1, endpoint, surrogate)
