"""Load the YAML config into a nested attribute-access object."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    return obj


def to_dict(ns) -> dict:
    if isinstance(ns, SimpleNamespace):
        return {k: to_dict(v) for k, v in vars(ns).items()}
    return ns


def load_config(path: str | Path | None = None) -> SimpleNamespace:
    with open(path or DEFAULT_CONFIG) as f:
        return _to_ns(yaml.safe_load(f))
