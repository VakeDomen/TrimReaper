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
    """

    def __init__(self, cfg: Config, split: str | None = None, holdout: bool = False):
        self.cfg = cfg
        split = split or cfg.data.split
        self._split = split
        self._holdout = holdout
        self._rng = torch.Generator().manual_seed(cfg.data.seed)
        self._lock = threading.Lock()
        self._docs: list[torch.Tensor] | None = None   # None => not loaded yet

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
        # materializes in memory; for large corpora prefer streaming with a
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
        # Shuffle + holdout split deterministically using the generator.
        perm = torch.randperm(len(docs), generator=self._rng)
        n_hold = max(1, int(len(docs) * 0.1)) if self._holdout else 0
        # For holdout=false we still need a deterministic ordering.
        if n_hold:
            self._holdout_idx = set(perm[:n_hold].tolist())
            self._docs = [d for i, d in enumerate(docs) if i not in self._holdout_idx]
        else:
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
        """Return the fixed VALIDATION set as a list of (fit_batch, seq_len) tensors.

        Used by the search to accept/ratchet records. Distinct windows are
        drawn sequentially from the seeded RNG, so calling ``test_batches()``
        afterwards yields different windows (disjoint by construction).
        """
        return self._fixed_windows(self.cfg.data.holdout_seq)

    def test_batches(self) -> list[torch.Tensor]:
        """Return the untouched TEST set as a list of (fit_batch, seq_len) tensors.

        Never used during search; evaluated only on recorded genomes for the
        final report so search can't overfit to it via repeated peeking.
        """
        return self._fixed_windows(self.cfg.data.test_seq)

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
