"""Model loading and MLP mask/rotation management.

Implements PLAN.md sections 2 and 4, with the corrected rotation semantics:

  - The experimental rotation acts on the POST-SwiGLU hidden activation, not on
    the raw gate/up weight rows. Composing the Givens rotations into an
    orthogonal matrix Q and applying ``x' = x @ Q`` right before ``down_proj``
    is EXACTLY function-preserving when ``down_proj``'s weight is counter-
    rotated by the same Q (``W_down @ Q``); with the mask disabled the MLP
    output is unchanged up to floating-point roundoff. This is a true basis
    rotation of the representation.
  - During search we do NOT physically delete channels; we simulate deletion
    by a forward pre-hook on each target MLP's ``down_proj`` that first rotates
    the hidden activation (``x @ Q``) then zeros the deleted (rotated) channels.
  - Physical compaction is a separate step (compaction.py) for final genomes.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from .config import Config
from .rotation import (
    DOWN,
    GATE,
    UP,
    PairRotation,
    make_orthogonal_matrix,
    rotate_down_weight,
)

MLP_MATRICES = (GATE, UP, DOWN)


def load_pruned_model(cfg: Config) -> PrunedModel:
    """Load a causal LM from HF and wrap it with rotation/mask machinery.

    Only the target layers' MLP weight copies are retained in memory in
    addition to the model itself (per PLAN.md section 14 memory strategy).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = cfg.resolve_device()
    dtype = cfg.resolve_dtype()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_id,
        dtype=dtype,
        device_map="auto" if device == "cuda" and torch.cuda.device_count() > 1 else device,
        low_cpu_mem_usage=cfg.model.low_cpu_mem_usage,
        trust_remote_code=True,
    )
    model.eval()
    return wrap_model(model, cfg, tokenizer)


def wrap_model(model, cfg: Config, tokenizer=None) -> PrunedModel:
    """Wrap an already-loaded model without HF redownload (used for smoke tests)."""
    pm = PrunedModel(model=model, config=cfg, tokenizer=tokenizer)
    wrap_target_mlps(pm, cfg.model.layers)
    return pm


def load_tiny_smoke_model(cfg: Config) -> PrunedModel:
    """Build a tiny randomly-initialized gated-MLP causal LM (Qwen3 classes).

    Used only as a fast CPU smoke test: same Qwen3MLP/decoder structure and the
    same wrapper code path, but tiny dimensions so a full GA run completes in
    seconds without a GPU or a large download.

    The 2-layer fixture only has layers 0 and 1; the target layer list is
    coerced to exist within [0, 1].
    """
    import torch
    from transformers import AutoModelForCausalLM, Qwen3Config

    conf = Qwen3Config(
        vocab_size=3200,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(conf)
    model.eval()
    # coerce target layers to fit within the 2-layer fixture
    cfg.model.layers = [l for l in cfg.model.layers if l < 2] or [0]
    return wrap_model(model, cfg)


def wrap_target_mlps(pm: PrunedModel, layers: list[int]) -> None:
    """Locate and snapshot the target MLPs (shared by load paths)."""
    model = pm.model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers_mod = model.model.layers
    else:
        raise NotImplementedError("Unsupported model layout: could not find .model.layers")

    for layer_idx in layers:
        layer = layers_mod[layer_idx]
        mlp = layer.mlp
        # guard: must be a gated MLP with gate/up/down projections
        for name in MLP_MATRICES:
            if not hasattr(mlp, name):
                raise NotImplementedError(f"MLP at layer {layer_idx} lacks {name}; not a gated MLP")
        pm.mlp_modules[layer_idx] = mlp
        pristine = {}
        for name in MLP_MATRICES:
            w = getattr(mlp, name).weight
            pristine[name] = w.data.detach().clone()
        pm.pristine[layer_idx] = pristine


@dataclass
class PrunedModel:
    """A wrapped model plus machinery to apply rotations and masks."""

    model: nn.Module
    config: Config
    tokenizer: Optional[object] = None
    # layer index -> pristine (untouched) weight tensors for the 3 matrices.
    # These live on the SAME device as the model (kept on GPU, no per-candidate
    # PCIe round-trips of 100+ MB of weights).
    pristine: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)
    # layer index -> module reference to the MLP (for hooks)
    mlp_modules: dict[int, nn.Module] = field(default_factory=dict)
    _hooks: list = field(default_factory=list)
    _masks: dict[int, torch.Tensor] = field(default_factory=dict)      # layer -> bool mask over ROTATED channels
    _rotations: dict[int, Optional[torch.Tensor]] = field(default_factory=dict)  # layer -> Q or None

    # ---- restoration ----------------------------------------------------
    def restore_all(self) -> None:
        """Restore every target MLP to its pristine weights and clear hooks."""
        for layer, mats in self.pristine.items():
            for name in MLP_MATRICES:
                getattr(self.mlp_modules[layer], name).weight.data.copy_(mats[name])
        self._rotations.clear()
        self._masks.clear()
        self._refresh_hooks()

    def clear_all_masks(self) -> None:
        """Remove every pruning mask (but keep any installed rotations)."""
        self._masks.clear()
        self._refresh_hooks()

    # ---- rotations / masks ---------------------------------------------
    def apply_rotations(self, layer: int, rots: list[PairRotation]) -> None:
        """Install a rotation Q (h' = h @ Q) + counter-rotated down_proj.

        ``gate``/``up`` stay pristine (rotating raw rows would break gated SiLU
        invariance). Instead Q acts on the hidden activation in the forward
        hook and ``down_proj`` is transformed as ``W_down @ Q``, making the
        MLP output exactly preserved when no channel is masked.
        """
        mlp = self.mlp_modules[layer]
        w = self.pristine[layer]
        width = w[UP].shape[0]
        dev = w[UP].device
        dt = w[UP].dtype
        if not rots:
            self._rotations[layer] = None
            mlp.down_proj.weight.data.copy_(w[DOWN])
        else:
            q = make_orthogonal_matrix(width, rots, device=dev, dtype=dt)
            self._rotations[layer] = q
            down_rot = rotate_down_weight(w[DOWN].to(dev), q)
            mlp.down_proj.weight.data.copy_(down_rot)
        self._refresh_hooks()

    def set_mask(self, layer: int, pruned: list[int], width: int | None = None) -> None:
        """Set the deletion mask for a layer (over ROTATED channels)."""
        if width is None:
            width = self.pristine[layer][UP].shape[0]
        mask = torch.ones(width, dtype=torch.bool, device=self.pristine[layer][UP].device)
        for c in pruned:
            mask[c] = False  # False -> rotated hidden channel is zeroed
        self._masks[layer] = mask
        self._refresh_hooks()

    def _refresh_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        for layer, mlp in self.mlp_modules.items():
            q = self._rotations.get(layer)
            mask = self._masks.get(layer)
            width = self.pristine[layer][UP].shape[0]

            def make_fn(q_ref, mask_ref, w_ref):
                def fn(mod, args):
                    x = args[0]
                    # x: (batch, seq, width) hidden input to down_proj.
                    if x.dim() < 2 or x.shape[-1] != w_ref:
                        return x
                    if q_ref is not None:
                        x = x @ q_ref.to(x.device)      # rotate hidden basis
                    if mask_ref is not None:
                        x = x.masked_fill(~mask_ref.to(x.device), 0.0)  # zero deleted channels
                    return x

                return fn

            self._hooks.append(mlp.down_proj.register_forward_pre_hook(make_fn(q, mask, width)))
