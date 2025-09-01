from __future__ import annotations

from dataclasses import replace

import pytest

from defi_security.config import get_profile
from defi_security.data import GRAPH_FEATURES, load_incidents
from defi_security.demo_data import generate_demo_data
from defi_security.modeling import chronological_split, run_experiment


def test_chronological_split_has_no_overlap(tmp_path) -> None:
    path = tmp_path / "incidents.csv"
    generate_demo_data().to_csv(path, index=False)
    dataframe = load_incidents(path)
    train, test = chronological_split(dataframe)
    assert train["incident_date"].max() <= test["incident_date"].min()
    assert len(train) == 135
    assert len(test) == 45


def test_demo_feature_sets_are_isolated(tmp_path) -> None:
    path = tmp_path / "incidents.csv"
    generate_demo_data().to_csv(path, index=False)
    dataframe = load_incidents(path)
    profile = replace(get_profile("demo"), data_path=path, output_dir=tmp_path / "artifacts")
    output = run_experiment(dataframe, profile)
    baseline = output.metrics["features"]["baseline"]
    included = output.metrics["features"]["graph_included"]
    assert not set(baseline).intersection(GRAPH_FEATURES)
    assert included == GRAPH_FEATURES
    assert output.metrics["target"]["threshold_scope"] == "training"
    assert output.metrics["temporal_cv"]["semantic"]["valid_folds"] >= 2


def test_predictive_profile_fails_closed_without_timestamped_graph_facts(tmp_path) -> None:
    path = tmp_path / "incidents.csv"
    dataframe = generate_demo_data()
    for feature in GRAPH_FEATURES:
        dataframe[f"{feature}_available_at"] = None
    dataframe.to_csv(path, index=False)
    loaded = load_incidents(path)
    profile = replace(get_profile("portfolio"), data_path=path)
    with pytest.raises(Exception, match="refusing predictive claims"):
        run_experiment(loaded, profile)

