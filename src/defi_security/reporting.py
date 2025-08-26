from __future__ import annotations

import json
from pathlib import Path

START_MARKER = "<!-- portfolio-metrics:start -->"
END_MARKER = "<!-- portfolio-metrics:end -->"


def render_metrics_table(metrics: dict) -> str:
    baseline = metrics["holdout"]["baseline"]
    semantic = metrics["holdout"]["semantic"]
    baseline_cv = metrics["temporal_cv"]["baseline"]
    semantic_cv = metrics["temporal_cv"]["semantic"]
    return "\n".join(
        [
            "| Model | Holdout AUC | F1 | Precision | Recall | Temporal CV AUC |",
            "|---|---:|---:|---:|---:|---:|",
            _row("Baseline", baseline, baseline_cv),
            _row("Graph-enriched", semantic, semantic_cv),
        ]
    )


def _row(name: str, holdout: dict, temporal_cv: dict) -> str:
    return (
        f"| {name} | {holdout['auc']:.3f} | {holdout['f1']:.3f} | "
        f"{holdout['precision']:.3f} | {holdout['recall']:.3f} | "
        f"{temporal_cv['auc_mean']:.3f} ± {temporal_cv['auc_std']:.3f} |"
    )


def validate_readme_metrics(readme: Path, metrics_path: Path) -> None:
    content = readme.read_text(encoding="utf-8")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    expected = render_metrics_table(metrics)
    if START_MARKER not in content or END_MARKER not in content:
        raise ValueError("README portfolio metric markers are missing")
    actual = content.split(START_MARKER, 1)[1].split(END_MARKER, 1)[0].strip()
    if actual != expected:
        raise ValueError("README portfolio metrics do not match artifacts/portfolio/metrics.json")

