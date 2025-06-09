from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ExperimentProfile:
    name: str
    data_path: Path
    output_dir: Path
    threshold_scope: str
    require_point_in_time: bool
    description: str


PROFILES = {
    "demo": ExperimentProfile(
        name="demo",
        data_path=PROJECT_ROOT / "data" / "sample" / "incidents.csv",
        output_dir=PROJECT_ROOT / "artifacts" / "demo",
        threshold_scope="training",
        require_point_in_time=True,
        description="Explicitly synthetic fixture for a local end-to-end smoke test.",
    ),
    "portfolio": ExperimentProfile(
        name="portfolio",
        data_path=PROJECT_ROOT / "data" / "processed" / "incidents.csv",
        output_dir=PROJECT_ROOT / "artifacts" / "portfolio",
        threshold_scope="training",
        require_point_in_time=True,
        description="Leakage-aware rerun using only features available by incident time.",
    ),
    "conference": ExperimentProfile(
        name="conference",
        data_path=PROJECT_ROOT / "data" / "conference" / "incidents.csv",
        output_dir=PROJECT_ROOT / "artifacts" / "conference",
        threshold_scope="all",
        require_point_in_time=False,
        description="Archival reproduction profile matching the documented 2025 paper setup.",
    ),
}


def get_profile(name: str) -> ExperimentProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown profile {name!r}; choose one of: {choices}") from exc

