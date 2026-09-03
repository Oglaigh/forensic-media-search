import os
from pathlib import Path
from types import SimpleNamespace

import console_ui


def test_summary_box_respects_terminal_width_and_wraps_values(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        console_ui.shutil,
        "get_terminal_size",
        lambda fallback: os.terminal_size((62, 24)),
    )
    config = SimpleNamespace(
        output_directory=Path("C:/salida/con/un/directorio/muy/extenso/para/el/reporte"),
        device="cuda",
        batch_size=64,
        top_k=5000,
        siglip_model="google/siglip2-base-patch16-224",
        clip_model="ViT-B/32",
    )
    console_ui.show_summary(
        config,
        Path("C:/evidencia/con/una/ruta/muy/extensa/que/debe/ajustarse"),
        ["consulta visual muy extensa " * 4],
    )
    lines = capsys.readouterr().out.splitlines()
    box_lines = [line for line in lines if line.startswith("  ")]
    assert box_lines
    assert all(len(line) == 62 for line in box_lines)
    assert box_lines[0].endswith("┐")
    assert box_lines[-1].endswith("┘")
