from __future__ import annotations

from typing import Any

import pandas as pd

from defi_security.config import ExperimentProfile
from defi_security.modeling import run_experiment


def threshold_sensitivity(
    dataframe: pd.DataFrame,
    profile: ExperimentProfile,
    *,
    quantiles: tuple[float, ...] = (0.70, 0.75, 0.80),
    seed: int = 42,
) -> dict[str, Any]:
    results: dict[str, Any] = {"schema_version": 1, "profile": profile.name, "runs": {}}
    for quantile in quantiles:
        output = run_experiment(
            dataframe,
            profile,
            seed=seed,
            severity_quantile=quantile,
        )
        results["runs"][f"q{int(quantile * 100)}"] = {
            "threshold_usd": output.metrics["target"]["threshold_usd"],
            "baseline_auc": output.metrics["holdout"]["baseline"]["auc"],
            "semantic_auc": output.metrics["holdout"]["semantic"]["auc"],
            "relative_auc_change_pct": output.metrics["holdout"][
                "relative_auc_change_pct"
            ],
        }
    return results

