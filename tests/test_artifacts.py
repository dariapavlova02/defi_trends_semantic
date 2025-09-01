from __future__ import annotations

import json
from dataclasses import replace

from defi_security.artifacts import write_artifacts
from defi_security.config import get_profile
from defi_security.data import load_incidents
from defi_security.demo_data import generate_demo_data
from defi_security.modeling import run_experiment


def test_artifacts_match_experiment_output(tmp_path) -> None:
    data_path = tmp_path / "incidents.csv"
    generate_demo_data().to_csv(data_path, index=False)
    profile = replace(get_profile("demo"), data_path=data_path, output_dir=tmp_path / "out")
    output = run_experiment(load_incidents(data_path), profile)
    write_artifacts(output, profile.output_dir, data_path)
    written = json.loads((profile.output_dir / "metrics.json").read_text())
    assert written == output.metrics
    assert (profile.output_dir / "model_comparison.png").stat().st_size > 10_000
    assert (profile.output_dir / "roc_curves.png").stat().st_size > 10_000

