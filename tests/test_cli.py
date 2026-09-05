import importlib
from pathlib import Path

import pytest

import forensic_media_search.cli as cli_module
from forensic_media_search.cli import build_parser, validate_args


def test_model_revisions_are_loaded_from_dotenv(tmp_path: Path, monkeypatch) -> None:
    with monkeypatch.context() as patch:
        patch.chdir(tmp_path)
        patch.delenv("SIGLIP2_REVISION", raising=False)
        patch.delenv("OPENAI_CLIP_REVISION", raising=False)
        (tmp_path / ".env").write_text(
            "SIGLIP2_REVISION=siglip-revision\n"
            "OPENAI_CLIP_REVISION=clip-revision\n",
            encoding="ascii",
        )

        reloaded_module = importlib.reload(cli_module)
        assert reloaded_module.SIGLIP2_REVISION == "siglip-revision"
        assert reloaded_module.OPENAI_CLIP_REVISION == "clip-revision"

    importlib.reload(cli_module)


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


def test_all_results_rejects_ensemble_options(tmp_path: Path, capsys) -> None:
    parser = build_parser()
    args = parser.parse_args([
        "--directory", str(tmp_path), "--query", "perro", "--device", "cpu",
        "--all-results", "--top-k", "10",
    ])
    with pytest.raises(SystemExit):
        validate_args(parser, args)
    assert "--all-results is incompatible" in capsys.readouterr().err


def test_cli_requires_paired_evaluator_options(tmp_path: Path, capsys) -> None:
    parser = build_parser()
    args = parser.parse_args([
        "--directory", str(tmp_path), "--query", "perro", "--device", "cpu",
        "--final-output", str(tmp_path.parent / "final.csv"),
    ])
    with pytest.raises(SystemExit):
        validate_args(parser, args)
    assert "must be provided together" in capsys.readouterr().err


def test_all_results_rejects_evaluator_options(tmp_path: Path, capsys) -> None:
    parser = build_parser()
    args = parser.parse_args([
        "--directory", str(tmp_path), "--query", "perro", "--device", "cpu",
        "--all-results", "--final-output", str(tmp_path.parent / "final.csv"),
        "--evaluator-top-k", "100",
    ])
    with pytest.raises(SystemExit):
        validate_args(parser, args)
    assert "--all-results is incompatible" in capsys.readouterr().err


def test_all_results_cli_never_constructs_clip(tmp_path: Path, monkeypatch) -> None:
    siglipd = object()
    sentinel_result = object()
    monkeypatch.setattr(cli_module, "SigLIP2Model", lambda *args, **kwargs: siglipd)
    monkeypatch.setattr(
        cli_module, "OpenAIClipModel",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("CLIP constructed")),
    )
    monkeypatch.setattr(cli_module, "run_all_results", lambda config, model: sentinel_result)
    monkeypatch.setattr(cli_module, "_print_all_results_startup", lambda args: None)
    monkeypatch.setattr(cli_module, "_print_all_results_summary", lambda result, count: None)
    assert cli_module.main([
        "--directory", str(tmp_path), "--query", "perro", "--device", "cpu",
        "--all-results", "--output", str(tmp_path.parent / "all.csv"),
    ]) == 0
