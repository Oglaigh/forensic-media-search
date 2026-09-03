from pathlib import Path
from types import SimpleNamespace

import pytest

import console


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "DOCKER_IMAGE": "forensic-media-search:dev",
        "OUTPUT_DIRECTORY": str(tmp_path / "output"),
        "CLIP_CACHE_DIRECTORY": str(tmp_path / "clip"),
        "HF_CACHE_DIRECTORY": str(tmp_path / "hf"),
        "TOP_K": "5000", "BATCH_SIZE": "64",
        "SIGLIP_MODEL": "google/siglip2-base-patch16-224",
        "CLIP_MODEL": "ViT-B/32", "DEVICE": "cuda",
        "RRF_CONSTANT": "60", "MAX_IMAGES": "", "REPORT_PREFIX": "report",
    }


def test_prompt_directory_retries_until_existing(tmp_path: Path, capsys) -> None:
    answers = iter([str(tmp_path / "missing"), f'"{tmp_path}"'])
    assert console.prompt_evidence_directory(lambda _: next(answers)) == tmp_path.resolve()
    assert "no existe" in capsys.readouterr().out


def test_prompt_queries_requires_one_and_stops_with_q(capsys) -> None:
    answers = iter([":q", "perro rojo", "robot", ":q"])
    assert console.prompt_queries(lambda _: next(answers)) == ["perro rojo", "robot"]
    assert "al menos una" in capsys.readouterr().out


def test_docker_command_mounts_evidence_read_only_and_appends_queries(tmp_path: Path) -> None:
    config = console.LauncherConfig.from_env(_env(tmp_path))
    command = console.build_docker_command(
        config, tmp_path.resolve(), ["perro", "robot"], "report.csv"
    )
    assert f"type=bind,source={tmp_path.resolve()},target=/evidence,readonly" in command
    assert command.count("--query") == 2
    assert command[-4:] == ["--query", "perro", "--query", "robot"]


def test_docker_command_adds_evaluator_arguments_and_container_output_path(
    tmp_path: Path,
) -> None:
    config = console.LauncherConfig.from_env(_env(tmp_path))
    final_output = config.output_directory / "review" / "final.csv"
    evaluation = SimpleNamespace(
        top_k=300,
        rrf_constant=60,
        final_output=final_output,
    )

    command = console.build_docker_command(
        config,
        tmp_path.resolve(),
        ["perro"],
        "audit.csv",
        evaluation,
    )

    assert command[command.index("--evaluator-top-k") + 1] == "300"
    assert command[command.index("--evaluator-rrf-constant") + 1] == "60"
    assert command[command.index("--final-output") + 1] == "/output/review/final.csv"
    assert "--output" in command


def test_docker_command_rejects_final_output_outside_output_directory(
    tmp_path: Path,
) -> None:
    config = console.LauncherConfig.from_env(_env(tmp_path))
    evaluation = SimpleNamespace(
        top_k=300,
        rrf_constant=60,
        final_output=tmp_path / "outside.csv",
    )

    with pytest.raises(console.ConfigurationError, match="OUTPUT_DIRECTORY"):
        console.build_docker_command(
            config, tmp_path.resolve(), ["perro"], "audit.csv", evaluation
        )


def test_artifacts_inside_evidence_are_rejected(tmp_path: Path) -> None:
    values = _env(tmp_path)
    values["OUTPUT_DIRECTORY"] = str(tmp_path / "evidence" / "output")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    with pytest.raises(console.ConfigurationError, match="solaparse"):
        console.validate_artifact_directories(
            console.LauncherConfig.from_env(values), evidence.resolve()
        )


def test_writable_mount_ancestor_of_evidence_is_rejected(tmp_path: Path) -> None:
    evidence = tmp_path / "case" / "evidence"
    evidence.mkdir(parents=True)
    values = _env(tmp_path)
    values["OUTPUT_DIRECTORY"] = str(evidence.parent)

    with pytest.raises(console.ConfigurationError, match="solaparse"):
        console.validate_artifact_directories(
            console.LauncherConfig.from_env(values), evidence.resolve()
        )


def test_configuration_rejects_invalid_numeric_value(tmp_path: Path) -> None:
    values = _env(tmp_path)
    values["BATCH_SIZE"] = "0"
    with pytest.raises(console.ConfigurationError, match="mayor que cero"):
        console.LauncherConfig.from_env(values)


def test_evaluator_defaults_do_not_require_new_env_values(tmp_path: Path) -> None:
    config = console.LauncherConfig.from_env(_env(tmp_path))

    assert config.evaluator_top_k == 300
    assert config.evaluator_rrf_constant == 60


def test_evaluator_env_defaults_are_configurable(tmp_path: Path) -> None:
    values = _env(tmp_path)
    values["EVALUATOR_TOP_K"] = "125"
    values["EVALUATOR_RRF_CONSTANT"] = "42"

    config = console.LauncherConfig.from_env(values)

    assert config.evaluator_top_k == 125
    assert config.evaluator_rrf_constant == 42
