from __future__ import annotations

import shutil
from pathlib import Path


def main() -> None:
    source = Path("artifacts/portfolio")
    destination = Path("docs/assets")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "model_comparison.png", destination / "portfolio_model_comparison.png")
    shutil.copy2(source / "roc_curves.png", destination / "portfolio_roc_curves.png")
    print(f"Updated curated portfolio assets in {destination}")


if __name__ == "__main__":
    main()
