from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import console_ui


def test_progress_bar_renders_percentage_and_counts(monkeypatch) -> None:
    output = io.StringIO()
    monkeypatch.setattr(console_ui.sys, "stdout", output)
    console_ui.ProgressBar(width=10).update("siglip2", 25, 100)
    rendered = output.getvalue()
    assert "SigLIP2" in rendered
    assert "25%" in rendered
    assert "25/100" in rendered


def test_style_is_plain_when_output_is_not_a_terminal(monkeypatch) -> None:
    monkeypatch.setattr(console_ui.sys, "stdout", io.StringIO())
    assert console_ui.styled("texto", console_ui.CYAN) == "texto"


def test_confirmation_defaults_to_no() -> None:
    assert console_ui.confirm(lambda _: "") is False
    assert console_ui.confirm(lambda _: "s") is True


def _evaluator_config(output_directory: Path) -> SimpleNamespace:
    return SimpleNamespace(
        output_directory=output_directory,
        evaluator_top_k=300,
        evaluator_rrf_constant=60,
    )


def test_final_evaluation_defaults_to_enabled_and_uses_defaults(tmp_path: Path) -> None:
    output_directory = (tmp_path / "output").resolve()
    answers = iter(["", "", "", ""])

    result = console_ui.prompt_final_evaluation(
        _evaluator_config(output_directory),
        output_directory / "audit.csv",
        "report_final.csv",
        lambda _: next(answers),
    )

    assert result == console_ui.FinalEvaluationOptions(
        top_k=300,
        rrf_constant=60,
        final_output=output_directory / "report_final.csv",
    )


def test_final_evaluation_can_be_disabled_without_affecting_audit(
    tmp_path: Path,
) -> None:
    output_directory = (tmp_path / "output").resolve()

    result = console_ui.prompt_final_evaluation(
        _evaluator_config(output_directory),
        output_directory / "audit.csv",
        "report_final.csv",
        lambda _: "n",
    )

    assert result is None


def test_final_evaluation_reprompts_for_numbers_and_unsafe_paths(
    tmp_path: Path, capsys
) -> None:
    output_directory = (tmp_path / "output").resolve()
    audit_output = output_directory / "audit.csv"
    answers = iter(
        [
            "s",
            "abc",
            "0",
            "25",
            "-1",
            "12",
            str(tmp_path / "outside.csv"),
            str(audit_output),
            "not-a-csv.txt",
            "nested/final.csv",
        ]
    )

    result = console_ui.prompt_final_evaluation(
        _evaluator_config(output_directory),
        audit_output,
        "report_final.csv",
        lambda _: next(answers),
    )

    assert result == console_ui.FinalEvaluationOptions(
        top_k=25,
        rrf_constant=12,
        final_output=output_directory / "nested" / "final.csv",
    )
    rendered = capsys.readouterr().out
    assert "número entero" in rendered
    assert "mayor que cero" in rendered
    assert "dentro de OUTPUT_DIRECTORY" in rendered
    assert "diferente" in rendered
    assert "archivo CSV" in rendered
