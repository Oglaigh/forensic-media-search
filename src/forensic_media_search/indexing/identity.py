"""Traceability fingerprints for loaded model adapters."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from typing import Any

from .metadata import canonical_sha256


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _stable_object(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_stable_object(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _stable_object(item) for key, item in value.items()}
    fields = getattr(value, "__dict__", None)
    if isinstance(fields, dict):
        return {
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                key: _stable_object(item)
                for key, item in sorted(fields.items())
                if not key.startswith("_")
            },
        }
    return {"class": f"{type(value).__module__}.{type(value).__qualname__}", "value": str(value)}


def _numeric_policy(model: Any) -> dict[str, Any]:
    policy: dict[str, Any] = {"device": model.info.device, "dtype": model.info.dtype}
    try:
        import torch
        policy.update({
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        })
        if model.info.device.startswith("cuda") and torch.cuda.is_available():
            index = torch.device(model.info.device).index
            index = torch.cuda.current_device() if index is None else index
            properties = torch.cuda.get_device_properties(index)
            driver_getter = getattr(torch._C, "_cuda_getDriverVersion", None)
            policy.update({
                "device_index": index,
                "gpu_name": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "total_memory": properties.total_memory,
                "gpu_uuid": str(getattr(properties, "uuid", "")) or None,
                "driver_version": driver_getter() if driver_getter is not None else None,
            })
    except ImportError:
        pass
    return policy


def _weight_sha256(model: Any) -> str:
    revision = model.info.revision or ""
    marker = "weights_sha256="
    if marker in revision:
        return revision.split(marker, 1)[1].split(";", 1)[0]
    digest = hashlib.sha256()
    state = getattr(model, "_model", None)
    if state is None or not hasattr(state, "state_dict"):
        raise RuntimeError(f"cannot resolve effective weights for {model.model_id}")
    for name, tensor in sorted(state.state_dict().items()):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().to("cpu").contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def model_identity(model: Any) -> dict[str, Any]:
    processor = getattr(model, "_processor", None)
    preprocess = getattr(model, "_preprocess", None)
    if processor is not None and hasattr(processor, "to_dict"):
        tokenizer = getattr(processor, "tokenizer", None)
        backend = getattr(tokenizer, "backend_tokenizer", None)
        preprocessing: Any = {
            "processor": processor.to_dict(),
            "tokenizer_init": getattr(tokenizer, "init_kwargs", None),
            "tokenizer_vocab_sha256": canonical_sha256(tokenizer.get_vocab())
            if tokenizer is not None and hasattr(tokenizer, "get_vocab") else None,
            "backend_tokenizer_sha256": canonical_sha256(backend.to_str())
            if backend is not None and hasattr(backend, "to_str") else None,
        }
    elif preprocess is not None:
        preprocessing = _stable_object(preprocess)
    else:
        preprocessing = None
    text_assets: dict[str, Any] = {}
    if model.info.model_id == "clip":
        try:
            from clip.simple_tokenizer import default_bpe
            from pathlib import Path
            bpe_path = Path(default_bpe())
            if bpe_path.is_file():
                digest = hashlib.sha256(bpe_path.read_bytes()).hexdigest()
                text_assets["bpe_sha256"] = digest
        except (ImportError, OSError):
            if type(model).__module__.endswith("models.openai_clip"):
                raise RuntimeError("cannot authenticate the OpenAI CLIP BPE asset")
            text_assets["bpe_sha256"] = None
    config = getattr(getattr(model, "_model", None), "config", None)
    resolved_revision = getattr(config, "_commit_hash", None) or model.info.revision
    identity = {
        "model_id": model.info.model_id,
        "name": model.info.name,
        "revision": model.info.revision,
        "resolved_revision": resolved_revision,
        "adapter_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "adapter_contract": 1,
        "weights_sha256": _weight_sha256(model),
        "preprocessing_sha256": canonical_sha256(preprocessing),
        "text_assets": text_assets,
        "numeric_policy": _numeric_policy(model),
        "packages": {
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "openai-clip": _package_version("openai-clip"),
            "torchvision": _package_version("torchvision"),
            "pillow": _package_version("Pillow"),
        },
    }
    return identity


def compatible_identity(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    fields = (
        "model_id", "name", "revision", "resolved_revision", "adapter_class",
        "adapter_contract", "weights_sha256", "preprocessing_sha256",
        "text_assets", "numeric_policy", "packages",
    )
    differences = [field for field in fields if expected.get(field) != actual.get(field)]
    if differences:
        raise ValueError(
            f"model/index identity mismatch for {actual.get('model_id')}: "
            + ", ".join(differences)
        )
