"""Styled terminal experience and live Docker progress rendering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
from typing import Callable, Sequence


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[96m"
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
GRAY = "\033[90m"


def _colors_enabled() -> bool:
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def styled(text: str, *codes: str) -> str:
    return f"{''.join(codes)}{text}{RESET}" if _colors_enabled() else text


def rule(character: str = "─", omit: int = 0) -> str:
    width = min(78, max(44, shutil.get_terminal_size((78, 24)).columns - 2))
    return character * max(0, width - omit)


def configure_output_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def banner() -> None:
    print(styled("╭" + rule("─") + "╮", CYAN))
    title = "  FORENSIC MEDIA SEARCH"
    subtitle = "  Búsqueda semántica offline · revisión de candidatos"
    print(
        styled("│", CYAN)
        + styled(title, BOLD, CYAN)
        + styled(rule(" ", len(title)) + "│", CYAN)
    )
    print(
        styled("│", CYAN)
        + styled(subtitle, DIM)
        + styled(rule(" ", len(subtitle)) + "│", CYAN)
    )
    print(styled("╰" + rule("─") + "╯", CYAN))


def section(number: str, title: str, subtitle: str) -> None:
    print()
    print(styled(f" {number}  {title} ", BOLD, BLUE))
    print(styled(subtitle, DIM))


def prompt_directory(input_fn=input) -> Path:
    section("01", "EVIDENCIA", "Seleccione el directorio que se examinará en modo solo lectura.")
    while True:
        raw = input_fn(styled("  Ruta › ", BOLD)).strip()
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'"}:
            raw = raw[1:-1]
        candidate = Path(raw).expanduser()
        if candidate.is_dir():
            resolved = candidate.resolve(strict=True)
            print(styled(f"  ✓ Directorio validado: {resolved}", GREEN))
            return resolved
        print(styled("  ✕ La ruta no existe o no es un directorio.", RED))


def prompt_queries(input_fn=input) -> list[str]:
    section("02", "CONSULTAS", "Ingrese una consulta por línea. Use :q para finalizar.")
    queries: list[str] = []
    while True:
        query = input_fn(styled(f"  Query {len(queries) + 1:02d} › ", BOLD)).strip()
        if query.casefold() == ":q":
            if queries:
                print(styled(f"  ✓ {len(queries)} consulta(s) registrada(s).", GREEN))
                return queries
            print(styled("  ! Debe ingresar al menos una consulta.", YELLOW))
        elif query:
            queries.append(query)
            print(styled("    agregada · Enter para continuar · :q para finalizar", DIM))
        else:
            print(styled("  ! La consulta no puede estar vacía.", YELLOW))


@dataclass(frozen=True, slots=True)
class FinalEvaluationOptions:
    top_k: int
    rrf_constant: int
    final_output: Path


def _prompt_yes_default(input_fn: Callable[[str], str]) -> bool:
    while True:
        answer = input_fn(
            styled("  ¿Desea generar un informe final consolidado? [S/n] › ", BOLD)
        ).strip().casefold()
        if answer in {"", "s", "si", "sí"}:
            return True
        if answer in {"n", "no"}:
            return False
        print(styled("  ! Responda 's' para generar el informe o 'n' para omitirlo.", YELLOW))


def _prompt_integer(
    label: str,
    default: int,
    minimum: int,
    input_fn: Callable[[str], str],
) -> int:
    while True:
        raw = input_fn(styled(f"  {label} [{default}] › ", BOLD)).strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            print(styled("  ! Ingrese un número entero.", YELLOW))
            continue
        if value < minimum:
            condition = "mayor que cero" if minimum == 1 else "no negativo"
            print(styled(f"  ! El valor debe ser {condition}.", YELLOW))
            continue
        return value


def _strip_matching_quotes(raw: str) -> str:
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'"}:
        return raw[1:-1]
    return raw


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolve_final_output(raw: str, output_directory: Path, default_name: str) -> Path:
    cleaned = _strip_matching_quotes(raw.strip()) or default_name
    candidate = Path(cleaned).expanduser()
    if not candidate.is_absolute():
        candidate = output_directory / candidate
    return candidate.resolve(strict=False)


def prompt_final_evaluation(
    config: object,
    audit_output: Path,
    default_final_name: str,
    input_fn=input,
) -> FinalEvaluationOptions | None:
    section(
        "03",
        "EVALUACIÓN FINAL",
        "Consolide por archivo sin alterar el informe completo de auditoría.",
    )
    if not _prompt_yes_default(input_fn):
        print(styled("  · Se generará únicamente el informe de auditoría.", DIM))
        return None

    top_k = _prompt_integer(
        "Top-K por query para evaluación", config.evaluator_top_k, 1, input_fn
    )
    rrf_constant = _prompt_integer(
        "Constante RRF del Evaluator",
        config.evaluator_rrf_constant,
        0,
        input_fn,
    )
    output_directory = Path(config.output_directory).resolve(strict=False)
    while True:
        raw = input_fn(styled(f"  Ruta del informe final [{default_final_name}] › ", BOLD))
        final_output = _resolve_final_output(raw, output_directory, default_final_name)
        if not _is_within(final_output, output_directory):
            print(styled("  ! La ruta debe estar dentro de OUTPUT_DIRECTORY.", YELLOW))
            continue
        if final_output == audit_output.resolve(strict=False):
            print(styled("  ! El informe final debe ser diferente al de auditoría.", YELLOW))
            continue
        if final_output == output_directory or final_output.suffix.lower() != ".csv":
            print(styled("  ! La ruta final debe identificar un archivo CSV.", YELLOW))
            continue
        print(styled(f"  ✓ Informe final: {final_output}", GREEN))
        return FinalEvaluationOptions(top_k, rrf_constant, final_output)


def show_summary(
    config: object,
    evidence: Path,
    queries: Sequence[str],
    *,
    audit_output: Path | None = None,
    evaluation: FinalEvaluationOptions | None = None,
) -> None:
    section("04", "CONFIRMACIÓN", "Revise la configuración antes de iniciar el análisis.")
    rows = [
        ("Evidencia", str(evidence)),
        ("Salida", str(config.output_directory)),
        ("Dispositivo", config.device.upper()),
        ("Batch size", str(config.batch_size)),
        ("Top-K", f"{config.top_k} por modelo y consulta"),
        ("Modelos", f"{config.siglip_model} + {config.clip_model}"),
    ]
    if audit_output is not None:
        rows.append(("Auditoría", str(audit_output)))
    if evaluation is None:
        rows.append(("Informe final", "No solicitado"))
    else:
        rows.extend(
            [
                ("Evaluación", f"Top-K {evaluation.top_k} por query"),
                ("RRF evaluator", str(evaluation.rrf_constant)),
                ("Informe final", str(evaluation.final_output)),
            ]
        )
    box_width = len(rule(" ", omit=2))
    content_width = box_width - 2
    label_width = 13

    def box_line(text: str, rendered: str | None = None) -> None:
        visible_padding = " " * max(0, content_width - len(text))
        print(
            styled("  │", GRAY)
            + " "
            + (text if rendered is None else rendered)
            + visible_padding
            + " "
            + styled("│", GRAY)
        )

    horizontal = rule("─", omit=2)
    print(styled("  ┌" + horizontal + "┐", GRAY))
    value_width = content_width - label_width - 1
    for label, value in rows:
        wrapped = textwrap.wrap(
            value,
            width=value_width,
            break_long_words=True,
            break_on_hyphens=False,
        ) or [""]
        for line_number, value_line in enumerate(wrapped):
            row_label = label if line_number == 0 else ""
            raw = f"{row_label:<{label_width}} {value_line}"
            rendered = f"{styled(row_label.ljust(label_width), DIM)} {value_line}"
            box_line(raw, rendered)

    print(styled("  ├" + horizontal + "┤", GRAY))
    box_line("Consultas", styled("Consultas", DIM))
    query_width = content_width - 4
    for index, query in enumerate(queries, 1):
        wrapped = textwrap.wrap(
            query,
            width=query_width,
            break_long_words=True,
            break_on_hyphens=False,
        ) or [""]
        for line_number, query_line in enumerate(wrapped):
            prefix = f"{index:02d}  " if line_number == 0 else " " * 4
            rendered_prefix = (
                f"{styled(f'{index:02d}', CYAN)}  "
                if line_number == 0
                else prefix
            )
            box_line(prefix + query_line, rendered_prefix + query_line)
    print(styled("  └" + horizontal + "┘", GRAY))
def confirm(input_fn=input) -> bool:
    while True:
        answer = input_fn(styled("  ¿Iniciar búsqueda? [s/N] › ", BOLD, YELLOW)).strip().casefold()
        if answer in {"s", "si", "sí"}:
            return True
        if answer in {"", "n", "no"}:
            return False
        print(styled("  ! Responda 's' para confirmar o 'n' para cancelar.", YELLOW))


class ProgressBar:
    def __init__(self, width: int = 32) -> None:
        self.width = width
        self.active = False
        self.phase = ""
        self.current = 0
        self.total = 0

    def update(self, phase: str, current: int = 0, total: int = 0) -> None:
        ratio = min(1.0, current / total) if total else 0.0
        filled = round(self.width * ratio)
        bar = "█" * filled + "░" * (self.width - filled)
        percent = round(ratio * 100)
        label = {
            "discovery": "Descubrimiento",
            "siglip2": "SigLIP2",
            "clip": "OpenAI CLIP",
        }.get(phase, phase)
        line = f"  {label:<14} {styled(bar, CYAN)} {percent:3d}%  {current:,}/{total:,}"
        if sys.stdout.isatty():
            print("\r\033[2K" + line, end="", flush=True)
        else:
            print(line)
        self.active = True
        self.phase = phase
        self.current = current
        self.total = total

    def is_complete(self, phase: str) -> bool:
        return self.phase == phase and self.total > 0 and self.current >= self.total

    def finish_line(self) -> None:
        if self.active and sys.stdout.isatty():
            print()
        self.active = False

    def clear(self) -> None:
        if self.active and sys.stdout.isatty():
            print("\r\033[2K", end="", flush=True)
        self.active = False


def run_with_progress(command: Sequence[str]) -> int:
    section("05", "PROGRESO", "El avance representa archivos examinados por cada modelo.")
    progress = ProgressBar()
    completed_models: set[str] = set()
    discovered = 0
    progress.update("Preparando", 0, 0)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip("\r\n")
        if line.startswith("@@PROGRESS\t"):
            try:
                _, phase, current, total = line.split("\t")
                current_value = int(current)
                total_value = int(total)
                progress.update(phase, current_value, total_value)
                if phase == "discovery":
                    discovered = total_value
            except (ValueError, TypeError):
                progress.clear()
                print(styled(f"  {line}", DIM))
        elif line.startswith("@@MODEL_COMPLETE\t"):
            _, phase = line.split("\t", 1)
            if phase not in completed_models and progress.is_complete(phase):
                progress.finish_line()
                label = {
                    "siglip2": "SigLIP2",
                    "clip": "OpenAI CLIP",
                }.get(phase, phase)
                print(styled(f"  ✓ {label} completado correctamente.", GREEN))
                print()
                completed_models.add(phase)
        elif line.startswith("PASS: "):
            progress.update(line.removeprefix("PASS: "), 0, discovered)
        elif line:
            progress.clear()
            print(styled(f"  │ {line}", DIM))
    return_code = process.wait()
    progress.clear()
    return return_code


def run_interactive(
    config: object,
    validate_directories: Callable[[object, Path], None],
    build_command: Callable[
        [object, Path, list[str], str, FinalEvaluationOptions | None], list[str]
    ],
) -> int:
    configure_output_encoding()
    banner()
    evidence = prompt_directory()
    queries = prompt_queries()
    validate_directories(config, evidence)
    timestamp = datetime.now()
    report_name = f"{config.report_prefix}_{timestamp:%Y%m%d_%H%M%S}.csv"
    final_name = f"{config.report_prefix}_{timestamp:%Y%m%d_%H%M%S}_final.csv"
    audit_output = config.output_directory / report_name
    evaluation = prompt_final_evaluation(config, audit_output, final_name)
    show_summary(
        config,
        evidence,
        queries,
        audit_output=audit_output,
        evaluation=evaluation,
    )
    if not confirm():
        print(styled("\n  Operación cancelada. No se generaron artefactos.", YELLOW))
        return 0

    config.output_directory.mkdir(parents=True, exist_ok=True)
    config.clip_cache_directory.mkdir(parents=True, exist_ok=True)
    config.hf_cache_directory.mkdir(parents=True, exist_ok=True)
    if evaluation is not None:
        evaluation.final_output.parent.mkdir(parents=True, exist_ok=True)
    return_code = run_with_progress(
        build_command(config, evidence, queries, report_name, evaluation)
    )
    if return_code == 0:
        print(styled("\n  ✓ BÚSQUEDA FINALIZADA", BOLD, GREEN))
        print(f"  Auditoría: {styled(str(audit_output), CYAN)}")
        if evaluation is not None:
            print(f"  Informe final: {styled(str(evaluation.final_output), CYAN)}")
    else:
        print(styled(f"\n  ✕ La búsqueda terminó con código {return_code}.", BOLD, RED))
    return return_code
