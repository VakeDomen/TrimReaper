"""Behavioral evaluation: KL divergence vs the original model.

Implements PLAN.md sections 8, 9, 12:
  - Baseline logits computed ONCE per generation on the shared batch.
  - Each candidate's logits are compared to baseline via token-averaged KL
    divergence of the next-token distribution.
  - No text generation during fitness; deterministic forward passes with
    inference mode, no cache, no gradients.
  - A fixed holdout set (not used in evolution) validates new records.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import PrunedModel


@torch.inference_mode()
def next_token_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Return logits for the next token at every position.

    Shape (batch, seq_len, vocab). Uses use_cache=False and no gradients.
    """
    out = model(
        input_ids=input_ids.to(model.device),
        use_cache=False,
        return_dict=True,
    )
    return out.logits


def kl_divergence(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    """Token-averaged KL(softmax(a) || softmax(b)).

    ``a`` = original (reference) distribution, ``b`` = candidate.
    Returns a scalar float.
    """
    log_pa = F.log_softmax(logits_a, dim=-1)
    pa = F.softmax(logits_a, dim=-1)
    log_pb = F.log_softmax(logits_b, dim=-1)
    kl = (pa * (log_pa - log_pb)).sum(dim=-1)
    return float(kl.mean().item())


@torch.inference_mode()
def baseline_logits(pm: PrunedModel, batches: list[torch.Tensor], seq_len: int) -> list[torch.Tensor]:
    """Compute original-model (pristine) logits for a set of batches."""
    # ensure target MLPs are pristine AND unmasked for the reference
    pm.restore_all()
    pm.clear_all_masks()
    refs = [next_token_logits(pm.model, b[:, :seq_len]) for b in batches]
    return refs


def evaluate_candidate(
    pm: PrunedModel,
    batch: torch.Tensor,
    baseline: torch.Tensor,
    layer: int,
    rots: list,
    pruned: list[int],
    seq_len: int,
) -> float:
    """Evaluate one candidate on a batch: restore, rotate, mask, compare.

    Returns token-averaged KL vs the provided baseline logits.
    """
    pm.restore_all()
    pm.apply_rotations(layer, rots)
    pm.set_mask(layer, pruned)
    try:
        cand = next_token_logits(pm.model, batch[:, :seq_len])
    finally:
        pm.restore_all()
    return kl_divergence(baseline, cand)


def _stream_kl(
    pm: PrunedModel,
    holdout_batches: list[torch.Tensor],
    layer: int,
    rots: list,
    pruned: list[int],
    seq_len: int,
) -> float:
    """Compare a candidate against the pristine model, batch by batch.

    Computes the reference logits on the SAME batch immediately before the
    candidate, then reduces to KL and drops both tensors. This keeps at most
    two logits in memory at once instead of materializing every holdout
    reference (which OOMs the 23 GiB card for wikitext-sized logits).
    """
    pm.restore_all()
    kls: list[float] = []
    for batch in holdout_batches:
        # reference (pristine, unmasked) on this batch
        pm.restore_all()
        pm.clear_all_masks()
        ref = next_token_logits(pm.model, batch[:, :seq_len])
        # candidate (rotated + masked) on the same batch
        pm.restore_all()
        pm.apply_rotations(layer, rots)
        pm.set_mask(layer, pruned)
        cand = next_token_logits(pm.model, batch[:, :seq_len])
        kls.append(kl_divergence(ref, cand))
        del ref, cand
    pm.restore_all()
    return float(sum(kls) / len(kls)) if kls else float("nan")


def evaluate_candidate_holdout(
    pm: PrunedModel,
    holdout_batches: list[torch.Tensor],
    layer: int,
    rots: list,
    pruned: list[int],
    seq_len: int,
) -> float:
    """Evaluate a candidate on the fixed VALIDATION set.

    Computes reference logits from the pristine model then compares the
    rotated+masked candidate against them (PLAN.md section 12). Streams one
    batch at a time to bound GPU memory. Returns mean KL over the set.
    """
    return _stream_kl(pm, holdout_batches, layer, rots, pruned, seq_len)


def evaluate_genome_on_set(
    pm: PrunedModel,
    batches: list[torch.Tensor],
    layer: int,
    genome,
    seq_len: int,
) -> float:
    """Evaluate a recorded Genome (rotations + pruned) on an arbitrary set."""
    return _stream_kl(
        pm, batches, layer,
        list(genome.rotations),
        sorted(genome.pruned),
        seq_len,
    )
