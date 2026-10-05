"""Fast behavioral evaluation: cache the prefix, evaluate only the MLP tail.

Implements the performance plan for the target-layer MLP-pruning search:

  For a given fitness batch, ONLY the target layer's MLP ``down_proj`` path
  differs between candidate genomes.  Everything before it is identical across
  candidates and is computed ONCE and cached:

      layers 0..(t-1)  ->  target-layer attention + post-attn residual
                          target-layer gate/up + SwiGLU  ->  h (the hidden
                          input to down_proj)

  Then each candidate (or a BATCH of candidates) only runs:

      cached h
         -> apply Givens rotations Q
         -> zero the deleted (rotated) coordinates
         -> apply inverse rotations Q^T
         -> ordinary, UNTOUCHED down_proj
         + cached post-attention residual
         -> layers (t+1)..end -> final norm -> lm_head

  This relies on the algebraic identity (M = pruning mask over rotated coords)

      (h Q M) (W Q)^T  ==  (h Q M Q^T) W^T

  so we NEVER rotate the ``down_proj`` weights during search and never restore
  them: the model weights stay pristine the whole run.  It also lets us branch
  the cached ``h`` into several candidate tails and run one batched forward of
  the shared suffix (optimization #3).

  Correctness is guaranteed by construction: caching hooks capture the exact
  intermediate tensors from a pristine forward, and the tail forward calls the
  model's own layer modules, so no transformer internals are reimplemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from .model import PrunedModel
from .rotation import DOWN, make_orthogonal_matrix

_ATTRS = ("attention_mask", "position_ids", "position_embeddings")


@dataclass
class MLPCache:
    """Everything before the target layer's down_proj, captured from one
    pristine forward of the model on a single batch."""

    layer: int
    h_swiglu: torch.Tensor          # (B, S, width) post-SwiGLU hidden -> down_proj input
    post_attn_residual: torch.Tensor  # (B, S, H) added to the MLP output
    ref_logits: torch.Tensor        # (B, S, V) pristine logits for KL
    attention_mask: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    position_embeddings: tuple = field(default_factory=tuple)

    @property
    def width(self) -> int:
        return self.h_swiglu.shape[-1]

    @property
    def seq_len(self) -> int:
        return self.h_swiglu.shape[1]


@torch.inference_mode()
def build_cache(pm: PrunedModel, batch: torch.Tensor, layer: int, seq_len: int) -> MLPCache:
    """Run ONE pristine forward of the whole model on ``batch``, capturing the
    reusable prefix tensors and returning the reference logits alongside them.

    Exactly one full-model forward runs here (also producing the reference the
    KL is measured against); every subsequent candidate eval reuses the cache.
    """
    model = pm.model
    layers = model.model.layers
    target_layer = layers[layer]

    capture: dict = {}

    def pre_down(_mod, args):
        capture["h_swiglu"] = args[0].detach()

    def pre_attn(_mod, _args, kwargs):
        capture["attention_mask"] = kwargs.get("attention_mask")
        capture["position_ids"] = kwargs.get("position_ids")
        capture["position_embeddings"] = kwargs.get("position_embeddings")

    def pre_postnorm(_mod, args):
        capture["post_attn_residual"] = args[0].detach()

    hooks = [
        target_layer.mlp.down_proj.register_forward_pre_hook(pre_down),
        target_layer.self_attn.register_forward_pre_hook(pre_attn, with_kwargs=True),
        target_layer.post_attention_layernorm.register_forward_pre_hook(pre_postnorm),
    ]
    try:
        inp = batch[:, :seq_len].to(model.device)
        ref = model(input_ids=inp, use_cache=False, return_dict=True).logits
    finally:
        for h in hooks:
            h.remove()

    pe = capture.get("position_embeddings")
    return MLPCache(
        layer=layer,
        h_swiglu=capture["h_swiglu"],
        post_attn_residual=capture["post_attn_residual"],
        ref_logits=ref.detach(),
        attention_mask=capture.get("attention_mask"),
        position_ids=capture.get("position_ids"),
        position_embeddings=tuple(pe) if isinstance(pe, (tuple, list)) else (),
    )


@torch.inference_mode()
def _run_suffix(cache: MLPCache, pm: PrunedModel, hidden: torch.Tensor) -> torch.Tensor:
    """Run layers (t+1).. + final norm + lm_head on ``hidden`` (N*B, S, H)."""
    model = pm.model
    layers = model.model.layers
    t = cache.layer
    x = hidden
    for i in range(t + 1, len(layers)):
        x = layers[i](
            x,
            attention_mask=cache.attention_mask,
            position_ids=cache.position_ids,
            position_embeddings=cache.position_embeddings,
            use_cache=False,
        )
    x = model.model.norm(x)
    return model.lm_head(x)


def _genome_q_mask(pm: PrunedModel, layer: int, genome, width: int):
    """Return (Q_or_None, mask) for a genome. mask is a bool tensor of shape
    (width,), True=keep. Handles empty-rotation / no-rotation genomes."""
    w = pm.pristine[layer][DOWN].shape[1]
    if w != width:
        width = w
    dev = pm.pristine[layer][DOWN].device
    dt = pm.pristine[layer][DOWN].dtype
    rots = list(getattr(genome, "rotations", []) or [])
    if rots:
        Q = make_orthogonal_matrix(width, rots, device=dev, dtype=dt)
    else:
        Q = None
    mask = torch.ones(width, dtype=torch.bool, device=dev)
    for c in (genome.pruned or ()):
        mask[int(c)] = False
    return Q, mask


@torch.inference_mode()
def eval_genome_fast(cache: MLPCache, pm: PrunedModel, layer: int, genome, seq_len: int) -> float:
    """Evaluate a SINGLE genome against the cache (unbatched tail)."""
    from .evaluate import kl_divergence

    Q, mask = _genome_q_mask(pm, layer, genome, cache.width)
    h = cache.h_swiglu
    if Q is not None:
        h = h @ Q
    h = h.masked_fill(~mask.to(h.device), 0.0)
    if Q is not None:
        h = h @ Q.transpose(-1, -2)
    out = pm.mlp_modules[layer].down_proj(h) + cache.post_attn_residual
    logits = _run_suffix(cache, pm, out)
    return kl_divergence(cache.ref_logits, logits)


@torch.inference_mode()
def _tail_batch(
    cache: MLPCache, pm: PrunedModel, layer: int, genomes: list
) -> list[float]:
    """Evaluate a list of genomes against the cache in ONE tail forward.

    Concatenates the candidates' ``down_proj`` outputs along the batch dim and
    runs a single forward of the shared suffix (layers after t + norm +
    lm_head), so the GPU sees an N*B effective batch.  Returns one KL per
    genome.
    """
    from .evaluate import kl_divergence

    if not genomes:
        return []
    outs = []
    for g in genomes:
        Q, mask = _genome_q_mask(pm, layer, g, cache.width)
        h = cache.h_swiglu
        if Q is not None:
            h = h @ Q
        h = h.masked_fill(~mask.to(h.device), 0.0)
        if Q is not None:
            h = h @ Q.transpose(-1, -2)
        outs.append(pm.mlp_modules[layer].down_proj(h) + cache.post_attn_residual)
    stacked = torch.cat(outs, dim=0)               # (N*B, S, H)
    logits_all = _run_suffix(cache, pm, stacked)   # (N*B, S, V)
    b = cache.ref_logits.shape[0]
    kls = []
    ref = cache.ref_logits
    for n in range(len(genomes)):
        cand = logits_all[n * b : (n + 1) * b]
        kls.append(kl_divergence(ref, cand))
    return kls


@torch.inference_mode()
def _bytes_per_candidate(cache: MLPCache, pm: PrunedModel) -> int:
    """Estimated peak bytes contributed by ONE candidate's tail forward.

    The dominant allocation is the ``lm_head`` output ``(B, S, V)`` in the
    model dtype, plus a roughly-equal FP32 copy used by the KL softmax math.
    """
    V = pm.model.lm_head.weight.shape[0]
    B = cache.ref_logits.shape[0]
    S = cache.seq_len
    el = cache.ref_logits.element_size()
    return B * S * V * el * 2


def _auto_chunk(cache: MLPCache, pm: PrunedModel, fallback: int = 4) -> int:
    """Choose the largest candidate chunk whose tail forward fits in free VRAM.

    Uses only ~60% of currently-free memory as a headroom budget so the run
    does not OOM on the lm_head logits.  Falls back to ``fallback`` (a
    batch-4-safe default) when CUDA is unavailable or the estimate fails.
    """
    if not torch.cuda.is_available():
        return fallback
    try:
        free, _ = torch.cuda.mem_get_info()
    except Exception:
        return fallback
    bpc = _bytes_per_candidate(cache, pm)
    if bpc <= 0:
        return fallback
    budget = int(free * 0.6)
    n = max(1, budget // bpc)
    return max(1, min(n, 64))


@torch.inference_mode()
def eval_genomes_batched(
    cache: MLPCache, pm: PrunedModel, layer: int, genomes: list, seq_len: int,
    chunk: int | None = None,
) -> list[float]:
    """Evaluate a LIST of genomes against the cache using the batched tail.

    ``chunk`` bounds how many candidate tails share one tail forward (e.g. 1,
    2, 4, 8) to control GPU VRAM.  ``chunk=None`` batches ALL genomes in a
    single tail forward; ``chunk=0`` auto-sizes from free VRAM so the run never
    OOMs on the lm_head logits.  Results are identical regardless of chunk size
    (only the memory/throughput trade-off differs).  ``seq_len`` is accepted
    for interface compatibility.
    """
    if not genomes:
        return []
    if chunk == 0:
        chunk = _auto_chunk(cache, pm)
    if chunk is None or chunk >= len(genomes):
        return _tail_batch(cache, pm, layer, genomes)
    kls: list[float] = []
    for i in range(0, len(genomes), chunk):
        kls.extend(_tail_batch(cache, pm, layer, genomes[i : i + chunk]))
    return kls
