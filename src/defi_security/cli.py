from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from defi_security.artifacts import write_artifacts
from defi_security.config import PROFILES, get_profile
from defi_security.data import (
    load_incidents,
    normalise_source_export,
    validate_incidents,
    write_normalised_dataset,
)
from defi_security.modeling import run_experiment
from defi_security.neo4j import Neo4jClient
from defi_security.robustness import threshold_sensitivity
from defi_security.settings import Neo4jSettings


def _run(args: argparse.Namespace) -> None:
    profile = get_profile(args.profile)
    data_path = Path(args.data) if args.data else profile.data_path
    output_dir = Path(args.output) if args.output else profile.output_dir
    dataframe = load_incidents(data_path)
    output = run_experiment(dataframe, profile, seed=args.seed)
    robustness = (
        threshold_sensitivity(dataframe, profile, seed=args.seed) if args.robustness else None
    )
    write_artifacts(output, output_dir, data_path, robustness=robustness)
    baseline = output.metrics["holdout"]["baseline"]["auc"]
    semantic = output.metrics["holdout"]["semantic"]["auc"]
    print(f"Profile: {profile.name}")
    print(f"Task: {output.metrics['task_label']}")
    print(f"Baseline AUC: {baseline:.3f}")
    print(f"Graph-enriched AUC: {semantic:.3f}")
    print(f"Artifacts: {output_dir}")


def _data_build(args: argparse.Namespace) -> None:
    dataframe = normalise_source_export(Path(args.input))
    destination = Path(args.output)
    write_normalised_dataset(dataframe, destination)
    print(f"Prepared {len(dataframe):,} incidents at {destination}")


def _data_validate(args: argparse.Namespace) -> None:
    dataframe = load_incidents(Path(args.input))
    validate_incidents(dataframe)
    print(f"Valid dataset: {len(dataframe):,} rows")


async def _graph_load_async(args: argparse.Namespace) -> None:
    dataframe = load_incidents(Path(args.input))
    settings = Neo4jSettings.from_env()
    rows = []
    for row in dataframe.itertuples(index=False):
        rows.append(
            {
                "event_id": row.event_id,
                "incident_date": row.incident_date.isoformat(),
                "loss_usd": float(row.loss_usd),
                "incident_type": row.incident_type,
                "target_type": row.target_type,
                "protocol_id": row.protocol_id,
            }
        )
    client = await Neo4jClient.create(settings)
    async with client:
        processed = await client.upsert_incidents(rows)
    print(f"Processed {processed:,} incidents in Neo4j")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="defi-security",
        description="Reproducible DeFi incident severity experiments",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run an experiment profile")
    run.add_argument("--profile", choices=sorted(PROFILES), required=True)
    run.add_argument("--data", help="override the profile dataset")
    run.add_argument("--output", help="override the artifact directory")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--robustness", action="store_true")
    run.set_defaults(handler=_run)

    data = commands.add_parser("data", help="prepare or validate input data")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    build = data_commands.add_parser("build", help="normalise a private source export")
    build.add_argument("--input", required=True)
    build.add_argument("--output", default="data/processed/incidents.csv")
    build.set_defaults(handler=_data_build)
    validate = data_commands.add_parser("validate", help="validate the public schema")
    validate.add_argument("--input", required=True)
    validate.set_defaults(handler=_data_validate)

    graph = commands.add_parser("graph", help="load normalized incidents into Neo4j")
    graph_commands = graph.add_subparsers(dest="graph_command", required=True)
    load = graph_commands.add_parser("load", help="idempotently upsert incidents")
    load.add_argument("--input", default="data/processed/incidents.csv")
    load.set_defaults(handler=lambda args: asyncio.run(_graph_load_async(args)))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.handler(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

