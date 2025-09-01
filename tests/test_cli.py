from __future__ import annotations

import argparse
import json

import pytest

from defi_security import cli
from defi_security.config import get_profile
from defi_security.demo_data import generate_demo_data


def test_run_command_writes_robustness_artifacts(tmp_path) -> None:
    data_path = tmp_path / "incidents.csv"
    output_path = tmp_path / "artifacts"
    generate_demo_data().to_csv(data_path, index=False)
    args = argparse.Namespace(
        profile="demo",
        data=str(data_path),
        output=str(output_path),
        seed=42,
        robustness=True,
    )
    cli._run(args)
    assert (output_path / "metrics.json").exists()
    robustness = json.loads((output_path / "robustness.json").read_text())
    assert sorted(robustness["runs"]) == ["q70", "q75", "q80"]


def test_data_build_and_validate_commands(tmp_path) -> None:
    source = tmp_path / "source.csv"
    destination = tmp_path / "processed" / "incidents.csv"
    generate_demo_data(60).to_csv(source, index=False)
    cli._data_build(argparse.Namespace(input=str(source), output=str(destination)))
    cli._data_validate(argparse.Namespace(input=str(destination)))
    assert destination.exists()


def test_parser_exposes_public_commands() -> None:
    parser = cli.build_parser()
    run = parser.parse_args(["run", "--profile", "demo"])
    validate = parser.parse_args(["data", "validate", "--input", "sample.csv"])
    graph = parser.parse_args(["graph", "load"])
    assert run.handler is cli._run
    assert validate.handler is cli._data_validate
    assert callable(graph.handler)


def test_unknown_profile_has_actionable_error() -> None:
    with pytest.raises(ValueError, match="choose one of"):
        get_profile("missing")

