"""Data streaming for fitness / holdout evaluation.

Implements PLAN.md section 8:
  - An ordinary text corpus (WikiText-103 by default), represented as random
    contiguous token windows.
  - Each *generation* draws a fresh random batch, but EVERY candidate in that
    generation is evaluated on that SAME batch, so candidate comparisons are
    not dominated by dataset noise.
  - A separate fixed holdout set (PLAN.md section 12) never participates in
    evolution and is used to validate new records.
"""

from __future__ import annotations

import threading
from typing import Optional

import torch
from datasets import load_dataset

from .config import Config


class WikiTextStreamer:
    """Streams contiguous token windows from a HF text dataset.

    Randomly samples a start offset in a shuffled list of tokenized documents,
    then produces contiguous windows of ``seq_len`` tokens.

    The fitness pool uses ``cfg.data.split`` (normally ``train``). The fixed
    VALIDATION and TEST sets come from WikiText's OWN ``validation`` and
    ``test`` splits, so the three pools are genuinely disjoint corpora (fix:
    sampling random windows from one shared pool let validation/test overlap,
    letting the search leak into the test set via repeated peeking).
    """

    # name (used internally) -> wiki split id
    _FIXED_SPLITS = {"validation": "validation", "test": "test"}

    def __init__(self, cfg: Config, split: str | None = None, holdout: bool = False):
        self.cfg = cfg
        split = split or cfg.data.split
        self._split = split
        self._holdout = holdout
        self._rng = torch.Generator().manual_seed(cfg.data.seed)
        self._lock = threading.Lock()
        self._docs: list[torch.Tensor] | None = None   # None => not loaded yet
        self._sub: dict[str, "WikiTextStreamer"] = {}

    def _fixed_streamer(self, split: str) -> "WikiTextStreamer":
        """A dedicated streamer over a disjoint fixed wiki split."""
        if split not in self._sub:
            self._sub[split] = WikiTextStreamer(self.cfg, split=split)
        return self._sub[split]

    # -- lazy loading -----------------------------------------------------
    def _ensure_docs(self) -> None:
        if self._docs is not None:
            return
        ds = load_dataset(
            self.cfg.data.dataset,
            self.cfg.data.dataset_config,
            split=self._split.split(":")[0],
            streaming=(":stream" in self._split),
        )
        # Build a corpus list of tokenized documents. For non-streaming this
        # materializes in memory; for the fixed validation/test splits the
        # corpus is small, for large fitness corpora prefer streaming with a
        # cap via ds.skip(...).take(...).
        tok = self._tokenizer()

        def _ids(text: str) -> torch.Tensor:
            enc = tok(text, add_special_tokens=False)
            return torch.tensor(enc["input_ids"], dtype=torch.long)

        docs: list[torch.Tensor] = []
        cap = self.cfg.data.max_docs
        for i, example in enumerate(ds):
            text = example.get("text")
            items = [t for t in ((text,) if isinstance(text, str) else (text or ())) if t]
            for t in items:
                ids = _ids(t)
                if ids.numel() > 0:
                    docs.append(ids)
                    if cap and len(docs) >= cap:
                        break
            if cap and len(docs) >= cap:
                break
        self._docs = docs

    def _tokenizer(self):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.cfg.model.model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    # -- sampling ---------------------------------------------------------
    def _sample_window(self, docs: list[torch.Tensor], seq_len: int) -> torch.Tensor:
        n = len(docs)
        idx = int(torch.randint(0, n, (1,), generator=self._rng).item())
        doc = docs[idx]
        max_start = max(0, doc.shape[0] - seq_len)
        if max_start == 0:
            # Short doc: pad with the last token (buffering a window).
            window = torch.nn.functional.pad(doc[:seq_len], (0, max(0, seq_len - doc.shape[0])), value=int(doc[-1].item()) if doc.numel() else 0)
            return window[:seq_len]
        start = int(torch.randint(0, max_start + 1, (1,), generator=self._rng).item())
        return doc[start : start + seq_len]

    def batch(self, n_seq: int, seq_len: int) -> torch.Tensor:
        """Return a fresh random batch of shape (n_seq, seq_len)."""
        with self._lock:
            self._ensure_docs()
            docs = self._docs or [torch.zeros(seq_len, dtype=torch.long)]
            seqs = [self._sample_window(docs, seq_len) for _ in range(n_seq)]
        return torch.stack(seqs).long()

    def holdout_batches(self) -> list[torch.Tensor]:
        """Return the fixed VALIDATION set (from WikiText's 'validation' split).

        Used by the search to accept/ratchet records. Genuinely disjoint from
        the train-based fitness pool and from the test set.
        """
        return self._fixed_streamer("validation")._fixed_windows(self.cfg.data.holdout_seq)

    def test_batches(self) -> list[torch.Tensor]:
        """Return the untouched TEST set (from WikiText's 'test' split).

        Never used during search; evaluated only on recorded genomes for the
        final report.
        """
        return self._fixed_streamer("test")._fixed_windows(self.cfg.data.test_seq)

    def _fixed_windows(self, n: int) -> list[torch.Tensor]:
        self._ensure_docs()
        seqs = list(self._docs)
        taken: list[torch.Tensor] = []
        if len(seqs) == 0:
            return [torch.zeros((1, self.cfg.data.seq_len), dtype=torch.long)]
        for i in range(n):
            taken.append(self._sample_window(seqs, self.cfg.data.seq_len))
        bs = self.cfg.data.fit_batch
        batches = []
        for i in range(0, len(taken), bs):
            batches.append(torch.stack(taken[i : i + bs]).long())
        return batches


def make_batch_tensors(cfg: Config, streamer: WikiTextStreamer) -> torch.Tensor:
    """Convenience wrapper: return one fitness batch tensor."""
    return streamer.batch(cfg.data.fit_batch, cfg.data.seq_len)
