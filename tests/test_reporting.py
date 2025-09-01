from __future__ import annotations

import json

import pytest

from defi_security.reporting import render_metrics_table, validate_readme_metrics


def _metrics() -> dict:
    score = {"auc": 0.8, "f1": 0.7, "precision": 0.6, "recall": 0.5}
    cv = {"auc_mean": 0.75, "auc_std": 0.05}
    return {
        "holdout": {"baseline": score, "semantic": score},
        "temporal_cv": {"baseline": cv, "semantic": cv},
    }


def test_readme_metric_validation(tmp_path) -> None:
    metrics_path = tmp_path / "metrics.json"
    readme = tmp_path / "README.md"
    metrics_path.write_text(json.dumps(_metrics()))
    table = render_metrics_table(_metrics())
    readme.write_text(
        f"before\n<!-- portfolio-metrics:start -->\n{table}\n"
        "<!-- portfolio-metrics:end -->\nafter\n"
    )
    validate_readme_metrics(readme, metrics_path)
    readme.write_text(readme.read_text().replace("0.800", "0.801", 1))
    with pytest.raises(ValueError, match="do not match"):
        validate_readme_metrics(readme, metrics_path)

