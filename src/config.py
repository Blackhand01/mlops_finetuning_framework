# src/config.py

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

import yaml

# =============================================================================
# Dataclass helpers                                                            
# =============================================================================

@dataclass(frozen=True)
class DatasetPaths:
    """Absolute paths to the train / valid / eval JSONL files."""
    train: Path
    valid: Path
    eval: Path

    def as_dict(self) -> Dict[str, str]:
        return {k: str(v) for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class FineTuningConfig:
    """Aggregates every parameter required to start a fine-tuning job."""
    lang_pair: str
    method: str
    version: str
    suffix: str
    base_model: str
    hyperparameters: Dict[str, Any]
    data: DatasetPaths


# =============================================================================
# Defaults                                                                     
# =============================================================================

DEFAULT_HYPERPARAMS: Dict[str, Any] = {
    "n_epochs": 3,
    "batch_size": 1,
    "learning_rate_multiplier": 2,
    "seed": 2063707736,
}
DEFAULT_BASE_MODEL = "gpt-4o-2024-08-06"
DEFAULT_SUFFIX_TEMPLATE = "{lang_pair}-translator-{version}-{method}"

# Keywords → field mapping for dataset discovery
_FILE_KEYWORDS: Dict[str, tuple[str, ...]] = {
    "train": ("train", "training"),
    "valid": ("valid", "validation"),
    "eval": ("eval", "evaluation"),
}


# =============================================================================
# Auto-builder                                                                 
# =============================================================================

def _auto_version(method_dir: Path) -> str:
    """Return the highest `v*` sub-folder name (e.g. `v7`)."""
    versions = sorted(
        (p for p in method_dir.iterdir() if p.is_dir() and p.name.lstrip("v").isdigit()),
        key=lambda p: int(p.name.lstrip("v"))
    )
    if not versions:
        raise FileNotFoundError(f"No 'v*' directories found in {method_dir}")
    return versions[-1].name


def _find_file(dir_path: Path, keywords: tuple[str, ...]) -> Path:
    """Return the first .jsonl file whose stem contains any of the keywords."""
    pattern = re.compile(r"|".join(map(re.escape, keywords)), re.IGNORECASE)
    for fp in dir_path.iterdir():
        if fp.is_file() and fp.suffix.lower() == ".jsonl" and pattern.search(fp.stem):
            return fp
    raise FileNotFoundError(f"No .jsonl matching {keywords} in {dir_path}")


def _build_ft_config(
    base_dir: Path,
    *,
    manual_version: Optional[str] = None,
    base_model: str = DEFAULT_BASE_MODEL,
    hyperparameters: Optional[Dict[str, Any]] = None,
    suffix_template: str = DEFAULT_SUFFIX_TEMPLATE,
) -> FineTuningConfig:
    """Derive every parameter needed for fine-tuning starting from base_dir."""
    hyperparameters = hyperparameters or DEFAULT_HYPERPARAMS

    # Expected directory layout: …/<lang_pair>/3_fineTuning/<method>
    method = base_dir.name
    lang_pair = base_dir.parent.parent.parent.name
    version = manual_version or _auto_version(base_dir)

    vdir = base_dir / version
    data_paths = DatasetPaths(
        train=_find_file(vdir, _FILE_KEYWORDS["train"]),
        valid=_find_file(vdir, _FILE_KEYWORDS["valid"]),
        eval=_find_file(vdir, _FILE_KEYWORDS["eval"]),
    )

    suffix = suffix_template.format(
        lang_pair=lang_pair,
        version=version.lstrip("v"),
        method=method,
    )

    return FineTuningConfig(
        lang_pair=lang_pair,
        method=method,
        version=version,
        suffix=suffix,
        base_model=base_model,
        hyperparameters=hyperparameters,
        data=data_paths,
    )


# =============================================================================
# Unified loader                                                               
# =============================================================================

class ConfigLoader:
    """Load a YAML config file or auto-generate config from a data directory."""

    def __init__(self, config_path: str) -> None:
        self.path = Path(config_path).expanduser().resolve()

    def load(self) -> Dict[str, Any]:
        """Return a validated configuration dictionary."""
        # If it's a YAML file, read and validate it
        if self.path.is_file() and self.path.suffix.lower() in {".yaml", ".yml"}:
            cfg = self._read_yaml()
            self._validate_schema(cfg)
            return cfg

        # Otherwise, zero-config mode: directory contains the data
        ft_cfg = _build_ft_config(self.path)
        return self._package_auto_config(ft_cfg)

    def _read_yaml(self) -> Dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as fp:
            try:
                return yaml.safe_load(fp) or {}
            except yaml.YAMLError as exc:
                raise yaml.YAMLError(f"YAML parsing error: {exc}") from exc

    def _validate_schema(self, cfg: Dict[str, Any]) -> None:
        """Shallow structural checks for classic YAML mode."""
        expected = {"dataset", "fine_tuning", "evaluation"}
        missing = expected - cfg.keys()
        if missing:
            raise ValueError(f"Missing top-level keys: {missing}")

        ds = cfg["dataset"]
        if "base_dir" not in ds or not Path(ds["base_dir"]).is_dir():
            raise ValueError("dataset.base_dir missing or invalid")

        ft = cfg["fine_tuning"]
        # 'method' is no longer mandatory: we derive it from the path
        for key in ("enable", "base_model", "suffix_template", "hyperparameters"):
            if key not in ft:
                raise ValueError(f"fine_tuning.{key} missing")

        ev = cfg["evaluation"]
        for key in ("enable", "data_source_config_path", "testing_criteria_path", "data_source_run_path"):
            if key not in ev:
                raise ValueError(f"evaluation.{key} missing")

    def _package_auto_config(self, ft: FineTuningConfig) -> Dict[str, Any]:
        """Produce the dict expected by pipeline_automatic in zero-config mode,
        resolving evaluation config paths dynamically."""
        project_root = self.path.parent
        cfg_dir = project_root / "evaluation" / "config"

        return {
            "dataset": {"base_dir": str(self.path)},
            "fine_tuning": {
                "enable": True,
                "base_model": ft.base_model,
                "method": ft.method,
                "suffix_template": DEFAULT_SUFFIX_TEMPLATE,
                "hyperparameters": ft.hyperparameters,
            },
            "evaluation": {
                "enable": True,
                "data_source_config_path": str(cfg_dir / "data_source_config.json"),
                "testing_criteria_path":     str(cfg_dir / "testing_criteria.json"),
                "data_source_run_path":      str(cfg_dir / "data_source_run.json"),
            },
            "_autobuild": {
                "lang_pair": ft.lang_pair,
                "version":   ft.version,
                "suffix":    ft.suffix,
                "data":      ft.data.as_dict(),
            },
        }
