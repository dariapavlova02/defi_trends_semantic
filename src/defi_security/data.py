from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

TEMPORAL_FEATURES = ["year", "month", "day_of_week"]
CATEGORICAL_FEATURES = ["incident_type", "target_type", "chain"]
GRAPH_FEATURES = [
    "protocol_chains_count",
    "is_forked_from_parent",
    "parent_fork_children_count",
    "protocol_past_events_count",
]
REQUIRED_COLUMNS = [
    "event_id",
    "incident_date",
    "loss_usd",
    "incident_type",
    "target_type",
    "chain",
    "protocol_id",
]


class DataValidationError(ValueError):
    """Raised when experiment input does not meet the public schema."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_incidents(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. See docs/DATA.md for preparation instructions."
        )
    dataframe = pd.read_csv(path)
    validate_incidents(dataframe)
    dataframe = dataframe.copy()
    dataframe["incident_date"] = pd.to_datetime(dataframe["incident_date"], utc=True)
    dataframe["loss_usd"] = pd.to_numeric(dataframe["loss_usd"], errors="raise")
    for feature in GRAPH_FEATURES:
        if feature not in dataframe:
            dataframe[feature] = np.nan
        availability = f"{feature}_available_at"
        if availability not in dataframe:
            dataframe[availability] = pd.NaT
        dataframe[availability] = pd.to_datetime(dataframe[availability], utc=True)
    return dataframe.sort_values(["incident_date", "event_id"]).reset_index(drop=True)


def validate_incidents(dataframe: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(dataframe.columns))
    if missing:
        raise DataValidationError(f"Missing required columns: {', '.join(missing)}")
    if dataframe.empty:
        raise DataValidationError("Dataset is empty")
    if dataframe["event_id"].duplicated().any():
        raise DataValidationError("event_id must be unique")
    dates = pd.to_datetime(dataframe["incident_date"], errors="coerce", utc=True)
    losses = pd.to_numeric(dataframe["loss_usd"], errors="coerce")
    if dates.isna().any():
        raise DataValidationError("incident_date contains invalid or missing values")
    if losses.isna().any() or (losses < 0).any():
        raise DataValidationError("loss_usd must contain non-negative numbers")


def eligible_point_in_time_features(dataframe: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    eligible: list[str] = []
    excluded: dict[str, str] = {}
    event_dates = dataframe["incident_date"]
    for feature in GRAPH_FEATURES:
        values = dataframe[feature]
        availability = dataframe[f"{feature}_available_at"]
        populated = values.notna()
        if not populated.any():
            excluded[feature] = "feature has no values"
        elif availability[populated].isna().any():
            excluded[feature] = "missing availability timestamps"
        elif (availability[populated] > event_dates[populated]).any():
            excluded[feature] = "contains facts observed after the incident"
        else:
            eligible.append(feature)
    return eligible, excluded


def _slug(value: object) -> str:
    text = str(value or "unknown").strip().lower()
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-") or "unknown"


def _normalise_chain(value: object) -> str:
    if isinstance(value, list):
        return "|".join(sorted({str(item) for item in value})) or "unknown"
    return str(value or "unknown")


def _source_column(dataframe: pd.DataFrame, name: str, default: object) -> pd.Series:
    if name in dataframe:
        return dataframe[name]
    return pd.Series([default] * len(dataframe), index=dataframe.index)


def normalise_source_export(path: Path) -> pd.DataFrame:
    """Normalise a Rekt-style JSON export or an already tabular CSV.

    Redistribution is intentionally not performed here. The caller owns the source export and
    must follow the source terms documented in data/provenance.json.
    """
    if path.suffix.lower() == ".csv":
        dataframe = pd.read_csv(path)
        validate_incidents(dataframe)
        result = dataframe.copy()
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("data", {}).get("rekts", payload)
        if not isinstance(records, list):
            raise DataValidationError("JSON must contain a list or data.rekts list")
        raw = pd.json_normalize(records)
        project_name = _source_column(raw, "projectName", "unknown")
        protocol_id = _source_column(raw, "defillamaId", None).fillna(project_name)
        result = pd.DataFrame(
            {
                "event_id": _source_column(raw, "id", None),
                "incident_date": _source_column(raw, "date", None),
                "loss_usd": _source_column(raw, "fundsLost", None),
                "incident_type": _source_column(raw, "issueType", "unknown"),
                "target_type": _source_column(raw, "category", "unknown"),
                "chain": _source_column(raw, "chaindIds", "unknown").map(_normalise_chain),
                "protocol_id": protocol_id.map(_slug),
            }
        )
    else:
        raise DataValidationError("Input must be a .csv or .json file")

    result["incident_date"] = pd.to_datetime(result["incident_date"], utc=True)
    result["loss_usd"] = pd.to_numeric(result["loss_usd"], errors="coerce")
    result = result.dropna(subset=["event_id", "incident_date", "loss_usd"]).copy()
    result["event_id"] = result["event_id"].astype(str)
    result = result.drop_duplicates("event_id").sort_values(["incident_date", "event_id"])

    # This count only uses strictly earlier incidents for the same protocol and is therefore
    # available at prediction time. Other graph facts require timestamped source metadata.
    result["protocol_past_events_count"] = result.groupby("protocol_id").cumcount().astype(float)
    result["protocol_past_events_count_available_at"] = result["incident_date"]
    for feature in GRAPH_FEATURES[:-1]:
        if feature not in result:
            result[feature] = np.nan
        result[f"{feature}_available_at"] = pd.NaT
    validate_incidents(result)
    return result.reset_index(drop=True)


def write_normalised_dataset(dataframe: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
