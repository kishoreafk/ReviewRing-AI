"""Deterministic seeding and small runtime helpers.

Every stochastic component in the project routes through :func:`seed_everything`
so that a run manifest (config + seed) reproduces the same data pipeline and
model initialisation. Torch determinism is set for CPU reproducibility; CUDA
flags are also set when a GPU is present, at a possible speed cost.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch generators deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():  # pragma: no cover - CI has no GPU
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Byte-level SHA256 of a file, streamed so large archives are safe."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_json(obj: Any) -> str:
    """Canonical JSON string for hashing manifests (sorted keys, no spaces)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def load_config(path: str | Path) -> dict:
    """Load a YAML config file into a plain dict."""
    with open(path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} did not parse into a mapping")
    return cfg


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` returning a new dict."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifact_dir(root: str | Path, run_id: str) -> Path:
    """Create and return ``<root>/<run_id>``; refuse to overwrite silently."""
    run_dir = Path(root) / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory {run_dir} already exists and is not empty. "
            "Use a new run id to avoid overwriting another run."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: str | Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, default=str)


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
