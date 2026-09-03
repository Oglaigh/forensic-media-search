from __future__ import annotations

import io

import console_ui


class FakeProcess:
    def __init__(self, lines: list[str], return_code: int) -> None:
        self.stdout = iter(f"{line}\n" for line in lines)
        self.return_code = return_code

    def wait(self) -> int:
        return self.return_code


def _run(monkeypatch, lines: list[str], return_code: int) -> str:
    output = io.StringIO()
    monkeypatch.setattr(console_ui.sys, "stdout", output)
    monkeypatch.setattr(
        console_ui.subprocess,
        "Popen",
        lambda *args, **kwargs: FakeProcess(lines, return_code),
    )
    assert console_ui.run_with_progress(["docker"]) == return_code
    return output.getvalue()


def test_each_model_success_is_printed_once_after_100_percent(monkeypatch) -> None:
    output = _run(
        monkeypatch,
        [
            "@@PROGRESS\tdiscovery\t100\t100",
            "PASS: siglip2",
            "@@PROGRESS\tsiglip2\t100\t100",
            "@@MODEL_COMPLETE\tsiglip2",
            "@@MODEL_COMPLETE\tsiglip2",
            "PASS: clip",
            "@@PROGRESS\tclip\t100\t100",
            "@@MODEL_COMPLETE\tclip",
        ],
        0,
    )
    siglip_message = "✓ SigLIP2 completado correctamente."
    clip_message = "✓ OpenAI CLIP completado correctamente."
    assert output.count(siglip_message) == 1
    assert output.count(clip_message) == 1
    assert output.index("100%  100/100") < output.index(siglip_message)
    assert output.rindex("100%  100/100") < output.index(clip_message)


def test_success_is_not_printed_without_100_percent(monkeypatch) -> None:
    output = _run(
        monkeypatch,
        [
            "@@PROGRESS\tdiscovery\t100\t100",
            "PASS: siglip2",
            "@@PROGRESS\tsiglip2\t50\t100",
            "@@MODEL_COMPLETE\tsiglip2",
        ],
        1,
    )
    assert "SigLIP2 completado correctamente" not in output


def test_success_is_not_printed_when_model_fails(monkeypatch) -> None:
    output = _run(
        monkeypatch,
        [
            "@@PROGRESS\tdiscovery\t100\t100",
            "PASS: clip",
            "@@PROGRESS\tclip\t100\t100",
            "inference failed",
        ],
        1,
    )
    assert "OpenAI CLIP completado correctamente" not in output
