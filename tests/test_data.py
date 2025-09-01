from __future__ import annotations

import json

import pandas as pd
import pytest

from defi_security.data import (
    DataValidationError,
    eligible_point_in_time_features,
    load_incidents,
    normalise_source_export,
    validate_incidents,
)
from defi_security.demo_data import generate_demo_data


def test_validate_rejects_duplicate_ids() -> None:
    dataframe = generate_demo_data(60)
    dataframe.loc[1, "event_id"] = dataframe.loc[0, "event_id"]
    with pytest.raises(DataValidationError, match="event_id must be unique"):
        validate_incidents(dataframe)


def test_point_in_time_filter_excludes_future_fact(tmp_path) -> None:
    path = tmp_path / "incidents.csv"
    dataframe = generate_demo_data(60)
    dataframe.loc[0, "protocol_chains_count_available_at"] = "2035-01-01T00:00:00Z"
    dataframe.to_csv(path, index=False)
    loaded = load_incidents(path)
    eligible, excluded = eligible_point_in_time_features(loaded)
    assert "protocol_chains_count" not in eligible
    assert excluded["protocol_chains_count"] == "contains facts observed after the incident"


def test_normalise_json_computes_only_prior_incidents(tmp_path) -> None:
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "data": {
                    "rekts": [
                        {
                            "id": "a",
                            "date": "2022-01-01",
                            "fundsLost": 10,
                            "issueType": "oracle",
                            "category": "dex",
                            "chaindIds": [1],
                            "projectName": "Alpha",
                        },
                        {
                            "id": "b",
                            "date": "2022-02-01",
                            "fundsLost": 20,
                            "issueType": "oracle",
                            "category": "dex",
                            "chaindIds": [1],
                            "projectName": "Alpha",
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    dataframe = normalise_source_export(source)
    assert dataframe["protocol_id"].tolist() == ["alpha", "alpha"]
    assert dataframe["protocol_past_events_count"].tolist() == [0.0, 1.0]
    assert pd.isna(dataframe.loc[0, "protocol_chains_count_available_at"])

