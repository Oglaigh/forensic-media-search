"""Interactive Windows launcher for the forensic media search container."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"


class ConfigurationError(ValueError):
    pass


def load_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ConfigurationError(
            f"No se encontro {path}. Copie .env.example como .env y configure sus valores."
        )
    values: dict[str, str] = {}
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ConfigurationError(f"Linea {number} invalida en {path.name}: falta '='.")
        key, value = (part.strip() for part in line.split("=", 1))
        if not key:
            raise ConfigurationError(f"Linea {number} invalida en {path.name}: clave vacia.")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def _required(values: dict[str, str], key: str) -> str:
    value = values.get(key, "").strip()
    if not value:
        raise ConfigurationError(f"Falta configurar {key} en .env.")
    return value


def _integer(values: dict[str, str], key: str, *, minimum: int) -> int:
    raw = _required(values, key)
    try:
        value = int(raw)
    except ValueError as error:
        raise ConfigurationError(f"{key} debe ser un numero entero.") from error
    if value < minimum:
        qualifier = "mayor que cero" if minimum == 1 else "no negativo"
        raise ConfigurationError(f"{key} debe ser {qualifier}.")
    return value


def _optional_positive_int(values: dict[str, str], key: str) -> int | None:
    raw = values.get(key, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as error:
        raise ConfigurationError(f"{key} debe ser un entero o quedar vacio.") from error
    if value <= 0:
        raise ConfigurationError(f"{key} debe ser mayor que cero o quedar vacio.")
    return value


def _config_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve(strict=False)


@dataclass(frozen=True, slots=True)
class LauncherConfig:
    docker_image: str
    output_directory: Path
    clip_cache_directory: Path
    hf_cache_directory: Path
    top_k: int
    batch_size: int
    siglip_model: str
    clip_model: str
    device: str
    rrf_constant: int
    max_images: int | None
    report_prefix: str

    @classmethod
    def from_env(cls, values: dict[str, str]) -> "LauncherConfig":
        device = _required(values, "DEVICE").lower()
        if device not in {"cuda", "cpu"}:
            raise ConfigurationError("DEVICE debe ser 'cuda' o 'cpu'.")
        prefix = _required(values, "REPORT_PREFIX")
        if any(character in prefix for character in '<>:"/\\|?*'):
            raise ConfigurationError("REPORT_PREFIX contiene caracteres no validos.")
        return cls(
            docker_image=_required(values, "DOCKER_IMAGE"),
            output_directory=_config_path(_required(values, "OUTPUT_DIRECTORY")),
            clip_cache_directory=_config_path(_required(values, "CLIP_CACHE_DIRECTORY")),
            hf_cache_directory=_config_path(_required(values, "HF_CACHE_DIRECTORY")),
            top_k=_integer(values, "TOP_K", minimum=1),
            batch_size=_integer(values, "BATCH_SIZE", minimum=1),
            siglip_model=_required(values, "SIGLIP_MODEL"),
            clip_model=_required(values, "CLIP_MODEL"),
            device=device,
            rrf_constant=_integer(values, "RRF_CONSTANT", minimum=0),
            max_images=_optional_positive_int(values, "MAX_IMAGES"),
            report_prefix=prefix,
        )


def prompt_evidence_directory(input_fn=input) -> Path:
    while True:
        raw = input_fn("Directorio de evidencia: ").strip()
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'"}:
            raw = raw[1:-1]
        candidate = Path(raw).expanduser()
        if candidate.is_dir():
            return candidate.resolve(strict=True)
        print("El directorio no existe o no es un directorio. Intente nuevamente.")


def prompt_queries(input_fn=input) -> list[str]:
    print("Ingrese una consulta por linea. Escriba :q para finalizar.")
    queries: list[str] = []
    while True:
        query = input_fn(f"Query {len(queries) + 1}: ").strip()
        if query.casefold() == ":q":
            if queries:
                return queries
            print("Debe ingresar al menos una consulta antes de finalizar.")
        elif query:
            queries.append(query)
        else:
            print("La consulta no puede estar vacia. Ingrese texto o :q para finalizar.")


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_artifact_directories(config: LauncherConfig, evidence: Path) -> None:
    for label, path in (
        ("OUTPUT_DIRECTORY", config.output_directory),
        ("CLIP_CACHE_DIRECTORY", config.clip_cache_directory),
        ("HF_CACHE_DIRECTORY", config.hf_cache_directory),
    ):
        if _is_within(path, evidence):
            raise ConfigurationError(f"{label} debe estar fuera del directorio de evidencia.")


def print_confirmation(config: LauncherConfig, evidence: Path, queries: list[str]) -> None:
    print("\nResumen de la busqueda")
    print(f"  Evidencia  : {evidence}")
    print(f"  Salida     : {config.output_directory}")
    print(f"  Dispositivo: {config.device}")
    print(f"  Batch size : {config.batch_size}")
    print(f"  Top-K      : {config.top_k} por modelo y query")
    print("  Queries:")
    for query in queries:
        print(f"    - {query}")


def confirm(input_fn=input) -> bool:
    while True:
        answer = input_fn("\nConfirma iniciar la busqueda? [s/N]: ").strip().casefold()
        if answer in {"s", "si", "sí"}:
            return True
        if answer in {"", "n", "no"}:
            return False
        print("Respuesta invalida. Ingrese 's' para confirmar o 'n' para cancelar.")


def build_docker_command(
    config: LauncherConfig, evidence: Path, queries: list[str], report_name: str
) -> list[str]:
    command = ["docker", "run", "--rm", "--env", "PYTHONUNBUFFERED=1"]
    if config.device == "cuda":
        command += ["--gpus", "all"]
    command += [
        "--mount", f"type=bind,source={evidence},target=/evidence,readonly",
        "--mount", f"type=bind,source={config.output_directory},target=/output",
        "--mount", f"type=bind,source={config.clip_cache_directory},target=/root/.cache/clip",
        "--mount", f"type=bind,source={config.hf_cache_directory},target=/root/.cache/huggingface",
        config.docker_image,
        "--directory", "/evidence", "--display-root", str(evidence),
        "--top-k", str(config.top_k), "--batch-size", str(config.batch_size),
        "--siglip-model", config.siglip_model, "--clip-model", config.clip_model,
        "--device", config.device, "--rrf-constant", str(config.rrf_constant),
        "--output", f"/output/{report_name}",
    ]
    if config.max_images is not None:
        command += ["--max-images", str(config.max_images)]
    for query in queries:
        command += ["--query", query]
    return command


def main() -> int:
    try:
        from console_ui import run_interactive

        config = LauncherConfig.from_env(load_env(ENV_PATH))
        return run_interactive(config, validate_artifact_directories, build_docker_command)
    except ConfigurationError as error:
        print(f"Error de configuracion: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nBusqueda cancelada.")
        return 130
    except OSError as error:
        print(f"No se pudo iniciar Docker: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
