from pathlib import Path

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


def test_artifacts_inside_evidence_are_rejected(tmp_path: Path) -> None:
    values = _env(tmp_path)
    values["OUTPUT_DIRECTORY"] = str(tmp_path / "evidence" / "output")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    with pytest.raises(console.ConfigurationError, match="fuera"):
        console.validate_artifact_directories(
            console.LauncherConfig.from_env(values), evidence.resolve()
        )


def test_configuration_rejects_invalid_numeric_value(tmp_path: Path) -> None:
    values = _env(tmp_path)
    values["BATCH_SIZE"] = "0"
    with pytest.raises(console.ConfigurationError, match="mayor que cero"):
        console.LauncherConfig.from_env(values)
