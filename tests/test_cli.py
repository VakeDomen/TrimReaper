"""Unit tests for CLI override parsing and config resolution."""

from types import SimpleNamespace

from trimreaper.cli import _load_config, _parse_overrides, build_parser, main
from trimreaper.config import Config


def test_parse_overrides_accepts_dashed_and_plain():
    assert _parse_overrides(["--ga.population=16", "search.epsilon=0.05"]) == {
        "ga.population": "16",
        "search.epsilon": "0.05",
    }


def test_argparse_captures_dotted_overrides():
    parser = build_parser()
    argv = ["baselines", "--ga.population=16", "--search.epsilon=0.05"]
    # mirror main()'s preprocessing of --group.key=value tokens
    proc = [t[2:] if (t.startswith("--") and "." in t.split("=", 1)[0]) else t for t in argv]
    args = parser.parse_args(proc)
    assert args.command == "baselines"
    assert args.overrides == ["ga.population=16", "search.epsilon=0.05"]


def test_load_config_applies_typed_overrides():
    args = SimpleNamespace(
        config="configs/poc.yaml",
        layers="18",
        overrides=[
            "ga.population=16",
            "search.epsilon=0.05",
            "data.seq_len=128",
            "model.layers=[18]",
        ],
    )
    cfg = _load_config(args)
    assert cfg.ga.population == 16
    assert cfg.search.epsilon == 0.05
    assert cfg.data.seq_len == 128
    assert cfg.model.layers == [18]


def test_sweep_eps_runs_without_gpu(capsys):
    rc = main(["sweep-eps"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "0.001" in out and "0.05" in out
