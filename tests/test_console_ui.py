from __future__ import annotations

import io

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
