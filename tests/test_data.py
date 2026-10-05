"""Unit tests for the WikiTextStreamer (lazy load + tokenization + holdout)."""

import torch
from datasets import Dataset


def _make_streamer(tmp_path, monkeypatch):
    from trimreaper.config import Config
    from trimreaper.data import WikiTextStreamer
    import trimreaper.data as D

    def fake_load_dataset(*a, **k):
        return Dataset.from_dict(
            {"text": ["The quick brown fox jumps over the lazy dog. " * 50] * 20}
        )

    monkeypatch.setattr(D, "load_dataset", fake_load_dataset)

    cfg = Config.defaults()
    # a real, small tokenizer so the tokenization path is genuinely exercised
    cfg.model.model_id = "Qwen/Qwen3-0.6B-Base"
    cfg.data.seq_len = 32
    cfg.data.fit_batch = 2
    cfg.data.holdout_seq = 8
    return WikiTextStreamer(cfg)


def test_streamer_lazy_loads_and_produces_real_batches(tmp_path, monkeypatch):
    s = _make_streamer(tmp_path, monkeypatch)
    b = s.batch(2, 32)
    assert b.shape == (2, 32)
    assert b.dtype == torch.long
    # tokenized "quick brown fox..." text -> mostly non-zero tokens
    assert torch.count_nonzero(b) > 0
    b2 = s.batch(3, 32)
    assert b2.shape == (3, 32)


def test_streamer_holdout_batches(tmp_path, monkeypatch):
    s = _make_streamer(tmp_path, monkeypatch)
    hb = s.holdout_batches()
    assert len(hb) == 8 // 2  # holdout_seq=8, fit_batch=2
    for x in hb:
        assert x.shape == (2, 32)
