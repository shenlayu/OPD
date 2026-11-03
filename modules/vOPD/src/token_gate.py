"""Scheme A on dense vocabularies, with an optional separate stop channel.

p = student, q = teacher. beta = sg[1(max(q) >= tau_q) * (1-p(argmax(q)))].
The JSD convention matches progressive_jsd: alpha=1-beta (0 FKL, 1 RKL).
We use exact full-vocabulary KLs because both models already return dense logits.

Core ablations: clip=inf disables forward clipping (C_F = infinity);
fixed_beta replaces the gate with one constant beta at every position;
normalize=False drops the 1/(1-beta) prefactor of psi_beta, i.e. the token loss
becomes sg(1-beta) * D, which restores the c_t weight on the pull toward argmax(q).
"""

import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .token_progress import PROGRESS_STATS
from .token_response import (ResponseConfig, RESPONSE_STATS, adjust_gate,
                             response_surrogate, sequence_future_factors)


STAT_NAMES = (
    "beta", "gate_fraction", "q_max", "student_at_teacher_argmax", "fkl", "rkl",
    "forward_component", "reverse_component", "clip_fraction", "stop_loss",
    "student_stop_probability", "teacher_stop_probability", "tail_mass",
)
CE_STATS = ("ce_replacement_fraction", "ce_component", "ce_replaced_jsd_reverse")


def validate_ce(kind, split_stop, progress_enabled, weight):
    if weight is None:
        return
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("token_gate_ce_weight must be finite and nonnegative, or null")
    if kind != "token_jsd" or split_stop or progress_enabled:
        raise ValueError("CE replacement requires token_jsd without split-stop or Scheme B")


def validate_gate(kind, tau_q, split_stop, stop_ids, k_odds, clip):
    if kind not in {"token_mix", "token_jsd"} or not 0 <= tau_q <= 1:
        raise ValueError("Invalid token objective or tau_q")
    if type(split_stop) is not bool:
        raise ValueError("token_gate_stop must be boolean")
    if k_odds != 1:
        raise ValueError("This sweep implements k_odds=1 only")
    if not stop_ids or len(set(stop_ids)) != len(stop_ids) or any(type(i) is not int or i < 0 for i in stop_ids):
        raise ValueError("Explicit distinct stop IDs are required, including for monitoring")
    if clip is not None and (math.isnan(clip) or clip <= 0):
        raise ValueError("token_gate_clip must be positive (inf disables clipping), or null for initial p95 calibration")


def validate_ablation(kind, fixed_beta, normalize, clip, split_stop, progress_enabled, ce_weight, response_enabled):
    """Core ablations stay single-factor: no stop channel, Scheme B, CE or response sweep."""
    if fixed_beta is None and normalize is True:
        return
    if fixed_beta is not None and (type(fixed_beta) not in {int, float} or not 0 <= fixed_beta <= 1):
        raise ValueError("token_gate_beta must be null or a number in [0, 1]")
    if type(normalize) is not bool:
        raise ValueError("token_gate_normalize must be boolean")
    if fixed_beta is not None and clip is None:
        raise ValueError("Constant beta needs an explicit token_gate_clip; the p95 calibration uses the gate")
    if not normalize and (kind != "token_jsd" or fixed_beta is not None):
        raise ValueError("token_gate_normalize=false ablates gated token_jsd only")
    if split_stop or progress_enabled or ce_weight is not None or response_enabled:
        raise ValueError("Core ablations require no split-stop, Scheme B, CE or response sweep")


def token_terms(student, teacher, tau_q, kind, split_stop, stop_ids, clip=None, progress=None, ce_weight=None,
                response=None, sampled=None, future=None, fixed_beta=None, normalize=True):
    """Per-token loss and detached diagnostics; no gradient through the gate.

Multiple stop IDs form one binary event. Their relative probabilities within
that event are not separately supervised in the split-stop ablation.
"""
    dtype = torch.float64 if student.dtype == torch.float64 else torch.float32
    validate_ce(kind, split_stop, progress is not None, ce_weight)
    if ce_weight is not None and clip is None:
        raise ValueError("CE replacement requires a calibrated or explicit clip threshold")
    lp = F.log_softmax(student, dim=-1, dtype=dtype)
    lq = F.log_softmax(teacher.detach(), dim=-1, dtype=dtype)
    qmax_log, best = lq.max(-1)
    qmax = qmax_log.exp()
    log_c = lp.gather(-1, best.unsqueeze(-1)).squeeze(-1)
    c = log_c.exp()
    gate = qmax >= tau_q
    beta = (gate * (1 - c)).detach()
    if fixed_beta is not None:
        # GKD-style constant beta everywhere; tau_q and c_t no longer matter.
        gate = torch.ones_like(gate)
        beta = torch.full_like(beta, fixed_beta)
    response = ResponseConfig.parse(response)
    response_stats = None
    if response.enabled:
        response.validate_context(kind, split_stop, progress is not None, ce_weight, stop_ids)
        beta, response_stats = adjust_gate(lp.detach(), lq, best, qmax, beta, tau_q,
                                           stop_ids, response, sampled, future)
        gate = beta > 0
    bucket_weight = torch.ones_like(beta)
    observations = None
    if progress is not None:
        beta, bucket_weight, observations = progress.policy(lp.detach(), lq, best, beta)
    selector = torch.ones(lp.shape[-1], dtype=torch.bool, device=lp.device)
    selector[list(stop_ids)] = False
    # Separate logsumexp reductions keep both tails stable even near p_stop=1.
    pstop = torch.logsumexp(lp[..., list(stop_ids)], -1)
    qstop = torch.logsumexp(lq[..., list(stop_ids)], -1)
    pcont = torch.logsumexp(lp[..., selector], -1)
    qcont = torch.logsumexp(lq[..., selector], -1)
    stop = qstop.exp() * (qstop - pstop) + qcont.exp() * (qcont - pcont)
    tail_mass = (lp.exp() * (lq < math.log(1e-4))).sum(-1).detach()
    if split_stop:
        # Crucially, the content mass cannot push the stop logit upward.
        content_weight = pcont.exp().detach()
        lp = lp[..., selector] - pcont.unsqueeze(-1)
        lq = lq[..., selector] - qcont.unsqueeze(-1)
    else:
        content_weight = torch.ones_like(beta)
    p, q = lp.exp(), lq.exp()
    fkl = (q * (lq - lp)).sum(-1)
    rkl = (p * (lp - lq)).sum(-1)
    if kind == "token_mix":
        forward, reverse = beta * fkl, (1 - beta) * rkl
    elif kind == "token_jsd":
        # Interior floor follows the established FP32 normalized-JSD protocol.
        # Exact endpoints stay exact; neither is approximated with the floor.
        b = beta.clamp(1e-3, 1 - 1e-3)
        lm = torch.logaddexp(lq + torch.log1p(-b).unsqueeze(-1), lp + b.log().unsqueeze(-1))
        forward = (q * (lq - lm)).sum(-1) / b
        reverse = (p * (lp - lm)).sum(-1) / (1 - b)
        forward = torch.where(beta == 0, torch.zeros_like(fkl), torch.where(beta == 1, fkl, forward))
        reverse = torch.where(beta == 0, rkl, torch.where(beta == 1, torch.zeros_like(rkl), reverse))
    else:
        raise ValueError(kind)
    clipped = torch.zeros_like(gate)
    if clip is not None:
        clipped = forward > clip
        forward = forward.clamp_max(clip)
    if not normalize:
        # Clip first, exactly as in Scheme A, then remove the 1/(1-beta) prefactor.
        # beta=0 keeps exact RKL; beta is detached, so gradients scale by 1-beta.
        forward, reverse = (1 - beta) * forward, (1 - beta) * reverse
    ce_stats = None
    if ce_weight is not None:
        selected = (gate & clipped).detach()
        # Keep the clipped forward constant C_F. Only replace the surviving
        # student-side JSD term; every other position retains exact Scheme A.
        ce = -ce_weight * log_c
        old_reverse = torch.where(selected, reverse, torch.zeros_like(reverse))
        reverse = torch.where(selected, ce, reverse)
        ce_stats = torch.stack((selected.to(dtype),
                               torch.where(selected, ce, torch.zeros_like(ce)), old_reverse), -1).detach()
    # Scheme B never scales the separate Bernoulli stop loss.
    loss = bucket_weight * content_weight * (forward + reverse) + (stop if split_stop else 0)
    if response.enabled:
        surrogate = response_surrogate(lp, lq, beta, clipped, response.power)
        # Preserve meaningful, comparable JSD monitoring; gradient is the surrogate.
        loss = loss.detach() + (surrogate - surrogate.detach())
    stats = torch.stack((beta, gate.to(dtype), qmax, c, fkl, rkl, forward, reverse,
                         clipped.to(dtype), stop, pstop.exp(), qstop.exp(), tail_mass), -1).detach()
    if observations is not None:
        stats = torch.cat((stats, observations), -1)
    if ce_stats is not None:
        stats = torch.cat((stats, ce_stats), -1)
    if response_stats is not None:
        stats = torch.cat((stats, response_stats), -1)
    return loss, stats


@torch.no_grad()
def calibrate_clip(student, teacher, mask, tau_q, split_stop, stop_ids, chunk_size=128):
    """Global p95 of beta*FKL on gated tokens in the first training microbatch.

All ranks participate, so all accumulation slices use one frozen threshold.
The reference is the same KL-mixture term for both divergence families.
No extra model forward and no training RNG consumption are needed.
"""
    s = student.detach().reshape(-1, student.shape[-1])
    t = teacher.detach().reshape_as(s)
    valid = mask.reshape(-1).bool()
    parts = []
    for offset in range(0, s.shape[0], chunk_size):
        sl = slice(offset, offset + chunk_size)
        keep = valid[sl]
        if not keep.any():
            continue
        _, stats = token_terms(s[sl][keep], t[sl][keep], tau_q, "token_mix", split_stop, stop_ids)
        parts.append((stats[:, 0] * stats[:, 4])[stats[:, 1].bool()])
    values = torch.cat(parts).float() if parts else s.new_empty(0, dtype=torch.float32)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        dist = torch.distributed
        count = torch.tensor([values.numel()], device=s.device, dtype=torch.long)
        counts = [torch.zeros_like(count) for _ in range(dist.get_world_size())]
        dist.all_gather(counts, count)
        sizes = [int(n.item()) for n in counts]
        padded = F.pad(values, (0, max(max(sizes), 1) - values.numel()))
        gathered = [torch.empty_like(padded) for _ in sizes]
        dist.all_gather(gathered, padded)
        values = torch.cat([v[:n] for v, n in zip(gathered, sizes)])
    if not values.numel():
        raise ValueError("Initial clip calibration has no gated tokens; set an explicit token_gate_clip")
    if not torch.isfinite(values).all():
        raise ValueError("Non-finite initial clip calibration")
    return max(float(torch.quantile(values, .95).item()), 1e-6), values.numel()


def dense_token_gate_loss(student, teacher, mask, *, tau_q, kind, split_stop,
                         stop_ids, clip, token_chunk_size=128, progress=None, observe_progress=True,
                         ce_weight=None, response=None, sampled_ids=None, fixed_beta=None, normalize=True):
    """Mean of per-sequence means, preserving equal sequence weight.

Checkpoint each token chunk to bound FP32 vocabulary intermediates. Padding is
removed before log-softmax, including from calibration and monitoring.
"""
    if student.shape != teacher.shape or student.ndim != 3 or mask.shape != student.shape[:-1]:
        raise ValueError("Expected matching [batch, time, vocab] logits and [batch, time] mask")
    if token_chunk_size < 1 or not stop_ids or max(stop_ids) >= student.shape[-1] or len(stop_ids) >= student.shape[-1]:
        raise ValueError("Invalid chunk size or stop IDs")
    validate_gate(kind, tau_q, split_stop, stop_ids, 1, clip)
    validate_ce(kind, split_stop, progress is not None, ce_weight)
    response = ResponseConfig.parse(response)
    response.validate_context(kind, split_stop, progress is not None, ce_weight, stop_ids)
    validate_ablation(kind, fixed_beta, normalize, clip, split_stop, progress is not None, ce_weight,
                      response.enabled)
    if response.reflection_ids and max(response.reflection_ids) >= student.shape[-1]:
        raise ValueError("Reflection ID is outside the vocabulary")
    future = torch.ones_like(mask, dtype=student.dtype)
    if response.future_horizon:
        if sampled_ids is None or sampled_ids.shape != mask.shape:
            raise ValueError("Future response gating requires aligned sampled completion IDs")
        future = sequence_future_factors(student, teacher, mask, response, token_chunk_size)
    sampled = (sampled_ids if sampled_ids is not None else torch.zeros_like(mask, dtype=torch.long)).reshape(-1)
    future = future.reshape(-1)
    weights = mask.to(torch.float64 if student.dtype == torch.float64 else torch.float32)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1) / mask.shape[0]
    weights = weights.reshape(-1)
    s = student.reshape(-1, student.shape[-1])
    t = teacher.detach().reshape_as(s)
    total = student.new_zeros((), dtype=weights.dtype)
    stat_names = STAT_NAMES + (PROGRESS_STATS if progress is not None else ())
    stat_names += CE_STATS if ce_weight is not None else ()
    stat_names += RESPONSE_STATS if response.enabled else ()
    summary = total.new_zeros(len(stat_names))
    bins = total.new_zeros((8, 3))

    def chunk(a, b, w, y, f):
        losses, stats = token_terms(a, b, tau_q, kind, split_stop, stop_ids, clip, progress, ce_weight,
                                   response, y, f, fixed_beta, normalize)
        observations = stats[:, len(STAT_NAMES):len(STAT_NAMES) + len(PROGRESS_STATS)] if progress is not None else stats[:, :0]
        return (losses * w).sum(), (stats * w.unsqueeze(-1)).sum(0), stats[:, 10:12], observations

    for offset in range(0, s.shape[0], token_chunk_size):
        sl = slice(offset, offset + token_chunk_size)
        keep = weights[sl] > 0
        if not keep.any():
            continue
        args = (s[sl][keep], t[sl][keep], weights[sl][keep], sampled[sl][keep], future[sl][keep])
        if torch.is_grad_enabled() and student.requires_grad:
            part, stats, probs, observations = checkpoint(chunk, *args, use_reentrant=False)
        else:
            part, stats, probs, observations = chunk(*args)
        if progress is not None and observe_progress:
            progress.observe(observations)
        total = total + part
        summary += stats
        positions = torch.arange(offset, min(offset + token_chunk_size, s.shape[0]), device=s.device)[keep] % mask.shape[1]
        bucket = (positions // 256).clamp_max(7)
        bins.index_add_(0, bucket, torch.cat((probs, torch.ones_like(probs[:, :1])), -1))
    metrics = dict(zip(stat_names, summary.unbind()))
    metrics.pop("progress_bucket", None)  # Bucket IDs are not scalar training metrics.
    for b in range(8):
        for i, name in enumerate(("student", "teacher")):
            metrics[f"{name}_stop_pos_{b * 256}"] = bins[b, i] / bins[b, 2].clamp_min(1)
    metrics["clip_threshold"] = total.new_tensor(clip)
    return total, metrics
