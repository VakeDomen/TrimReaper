"""Model loading and MLP mask/rotation management.

Implements PLAN.md sections 2 and 4:
  - Load a HF causal LM (BF16) without keeping a second full model in memory.
  - Keep a pristine copy of each *target MLP's* three weight tensors so
    candidates can restore + apply rotations cheaply.
  - During search, do NOT physically delete channels. Instead simulate
    deletion via a forward pre-hook on each target MLP's ``down_proj`` that
    zeros the hidden-channel activations in place (hidden[channel] = 0).
  - Physical compaction is a separate step (compaction.py) for final genomes.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from .config import Config
from .rotation import DOWN, GATE, UP, PairRotation, apply_rotation_sequence

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
            pristine[name] = w.data.detach().clone().cpu()
        pm.pristine[layer_idx] = pristine


@dataclass
class PrunedModel:
    """A wrapped model plus machinery to apply rotations and masks."""

    model: nn.Module
    config: Config
    tokenizer: Optional[object] = None
    # layer index -> pristine (untouched) weight tensors for the 3 matrices
    pristine: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)
    # layer index -> module reference to the MLP (for hooks)
    mlp_modules: dict[int, nn.Module] = field(default_factory=dict)
    _hooks: list = field(default_factory=list)
    _masks: dict[int, torch.Tensor] = field(default_factory=dict)  # layer -> bool mask over channels

    def restore_all(self) -> None:
        """Restore every target MLP to its pristine weights."""
        for layer, mats in self.pristine.items():
            for name in MLP_MATRICES:
                mat = mats[name]
                getattr(self.mlp_modules[layer], name).weight.data.copy_(mat)

    def clear_all_masks(self) -> None:
        """Remove every pruning mask and its hooks (returns to unmasked)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._masks.clear()

    def apply_rotations(self, layer: int, rots: list[PairRotation]) -> None:
        """Apply a rotation sequence to one layer's MLP (must be pristine)."""
        mlp = self.mlp_modules[layer]
        params = {name: getattr(mlp, name).weight.data for name in MLP_MATRICES}
        apply_rotation_sequence(params, rots)

    def set_mask(self, layer: int, pruned: list[int], width: int | None = None) -> None:
        """Set the deletion mask for a layer to the (zeroed) pruned channels."""
        if width is None:
            width = self.pristine[layer][UP].shape[0]
        mask = torch.ones(width, dtype=torch.bool, device=self.model.device)
        for c in pruned:
            mask[c] = False  # False -> hidden channel is zeroed
        self._masks[layer] = mask
        self._update_hooks()

    def _update_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        for layer, mask in self._masks.items():
            mlp = self.mlp_modules[layer]

            def make_fn(mask_ref):
                def fn(mod, args):
                    x = args[0]
                    keep = mask_ref
                    # x: (batch, seq, intermediate) — zero deleted channels.
                    if x.shape[-1] != keep.shape[0]:
                        return x
                    x = x.masked_fill(~keep.to(x.device), 0.0)
                    return x

                return fn

            self._hooks.append(mlp.down_proj.register_forward_pre_hook(make_fn(mask)))
