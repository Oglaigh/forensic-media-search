from pathlib import Path

import pytest

from forensic_media_search.cli import build_parser, validate_args


def test_cli_top_k_means_per_model_and_query(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args([
        "--directory", str(tmp_path), "--query", "perro", "--query", "robot",
        "--top-k", "50", "--device", "cpu",
    ])
    validate_args(parser, args)
    assert args.top_k == 50
    assert args.query == ["perro", "robot"]


def test_legacy_percentage_flags_fail_with_migration_message(tmp_path: Path, capsys) -> None:
    parser = build_parser()
    args = parser.parse_args([
        "--directory", str(tmp_path), "--query", "perro", "--device", "cpu",
        "--min-percent", "75",
    ])
    with pytest.raises(SystemExit):
        validate_args(parser, args)
    assert "use per-query --top-k" in capsys.readouterr().err


def test_model_caches_inside_evidence_are_rejected(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))
    parser = build_parser()
    args = parser.parse_args([
        "--directory", str(tmp_path), "--query", "perro", "--device", "cpu",
    ])
    with pytest.raises(SystemExit):
        validate_args(parser, args)
    assert "HF_HOME must resolve outside" in capsys.readouterr().err
