import os

import console_ui


def _terminal_width(monkeypatch, columns: int) -> None:
    monkeypatch.setattr(
        console_ui.shutil, "get_terminal_size", lambda fallback: os.terminal_size((columns, 24))
    )


def test_rule_without_omit_preserves_available_width(monkeypatch) -> None:
    _terminal_width(monkeypatch, 62)
    assert console_ui.rule("─") == "─" * 60


def test_rule_omit_reduces_length_and_never_becomes_negative(monkeypatch) -> None:
    _terminal_width(monkeypatch, 62)
    assert len(console_ui.rule(" ", 10)) == len(console_ui.rule(" ")) - 10
    assert console_ui.rule(" ", 100) == ""


def test_rule_keeps_minimum_and_maximum_width(monkeypatch) -> None:
    _terminal_width(monkeypatch, 20)
    assert len(console_ui.rule()) == 44
    _terminal_width(monkeypatch, 200)
    assert len(console_ui.rule()) == 78


def test_banner_lines_have_equal_visible_width(monkeypatch, capsys) -> None:
    _terminal_width(monkeypatch, 62)
    console_ui.banner()
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4
    assert {len(line) for line in lines} == {62}
    assert all(line.endswith(("│", "╮", "╯")) for line in lines)
