from __future__ import annotations

import importlib.metadata
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from defi_security.config import PROJECT_ROOT
from defi_security.data import file_sha256
from defi_security.modeling import ExperimentOutput, roc_coordinates

COLORS = {"baseline": "#94A3B8", "semantic": "#14B8A6", "accent": "#0F172A"}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def _versions() -> dict[str, str]:
    return {
        package: importlib.metadata.version(package)
        for package in ("pandas", "scikit-learn", "lightgbm", "matplotlib")
    }


def _model_comparison(output: ExperimentOutput, destination: Path) -> None:
    baseline = output.metrics["holdout"]["baseline"]
    semantic = output.metrics["holdout"]["semantic"]
    labels = ["AUC", "F1", "Precision", "Recall"]
    keys = ["auc", "f1", "precision", "recall"]
    positions = np.arange(len(labels))
    width = 0.34
    figure, axis = plt.subplots(figsize=(9, 5.2))
    axis.bar(
        positions - width / 2,
        [baseline[key] for key in keys],
        width,
        label="Baseline",
        color=COLORS["baseline"],
    )
    axis.bar(
        positions + width / 2,
        [semantic[key] for key in keys],
        width,
        label="Graph-enriched",
        color=COLORS["semantic"],
    )
    axis.set_ylim(0, 1)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Score")
    axis.set_title("Chronological holdout performance")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _roc_plot(output: ExperimentOutput, destination: Path) -> None:
    baseline_x, baseline_y = roc_coordinates(output.y_test, output.baseline_probability)
    semantic_x, semantic_y = roc_coordinates(output.y_test, output.semantic_probability)
    baseline_auc = output.metrics["holdout"]["baseline"]["auc"]
    semantic_auc = output.metrics["holdout"]["semantic"]["auc"]
    figure, axis = plt.subplots(figsize=(6.4, 5.4))
    axis.plot(
        baseline_x,
        baseline_y,
        color=COLORS["baseline"],
        label=f"Baseline ({baseline_auc:.3f})",
    )
    axis.plot(
        semantic_x,
        semantic_y,
        color=COLORS["semantic"],
        label=f"Graph-enriched ({semantic_auc:.3f})",
    )
    axis.plot([0, 1], [0, 1], linestyle="--", color="#CBD5E1")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1))
    axis.set_title("ROC curves — chronological holdout")
    axis.legend(frameon=False, loc="lower right")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def write_artifacts(
    output: ExperimentOutput,
    output_dir: Path,
    data_path: Path,
    *,
    robustness: dict[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "metrics.json", output.metrics)
    resolved_data_path = data_path.resolve()
    try:
        display_data_path = str(resolved_data_path.relative_to(PROJECT_ROOT))
    except ValueError:
        display_data_path = data_path.name
    metadata = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "data_file": display_data_path,
        "data_sha256": file_sha256(data_path),
        "library_versions": _versions(),
    }
    _write_json(output_dir / "run_metadata.json", metadata)
    if robustness is not None:
        _write_json(output_dir / "robustness.json", robustness)
    _model_comparison(output, output_dir / "model_comparison.png")
    _roc_plot(output, output_dir / "roc_curves.png")
