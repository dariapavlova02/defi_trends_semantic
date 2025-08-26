from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    source = Path("artifacts/portfolio")
    destination = Path("docs/assets")
    metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "model_comparison.png", destination / "portfolio_model_comparison.png")
    shutil.copy2(source / "roc_curves.png", destination / "portfolio_roc_curves.png")

    baseline = metrics["holdout"]["baseline"]["auc"]
    semantic = metrics["holdout"]["semantic"]["auc"]
    figure = plt.figure(figsize=(12.8, 6.4), facecolor="#07111F")
    axis = figure.add_axes((0, 0, 1, 1))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_axis_off()
    axis.text(
        0.07,
        0.72,
        "DEFI SECURITY\nKNOWLEDGE GRAPH",
        color="#F8FAFC",
        fontsize=35,
        fontweight="bold",
        va="center",
        linespacing=1.1,
    )
    axis.text(
        0.07,
        0.35,
        "Leakage-aware graph feature engineering\nfor incident severity analysis",
        color="#94A3B8",
        fontsize=16,
        va="center",
        linespacing=1.5,
    )
    axis.text(
        0.07,
        0.12,
        "SIKDD 2025  •  PYTHON  •  NEO4J  •  LIGHTGBM",
        color="#2DD4BF",
        fontsize=12,
    )
    axis.plot([0.67, 0.78, 0.89], [0.65, 0.79, 0.57], color="#2DD4BF", linewidth=2.5)
    axis.plot([0.67, 0.78, 0.89, 0.79], [0.65, 0.79, 0.57, 0.34], color="#334155", linewidth=1.5)
    axis.scatter(
        [0.67, 0.78, 0.89, 0.79],
        [0.65, 0.79, 0.57, 0.34],
        s=[800, 1100, 700, 900],
        color=["#14B8A6", "#0EA5E9", "#8B5CF6", "#F59E0B"],
        alpha=0.9,
    )
    axis.text(
        0.78,
        0.13,
        f"Corrected holdout\nAUC  {baseline:.3f}  →  {semantic:.3f}",
        color="#E2E8F0",
        fontsize=14,
        ha="center",
        linespacing=1.5,
    )
    figure.savefig(destination / "hero.png", dpi=100, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(f"Updated curated portfolio assets in {destination}")


if __name__ == "__main__":
    main()
