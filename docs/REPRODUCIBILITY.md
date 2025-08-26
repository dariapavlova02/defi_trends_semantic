# Reproducibility

## Environment

The lock file targets Python 3.11 and 3.12. Install all runtime, graph, and development dependencies:

```bash
uv sync --all-extras --locked
```

## Demo profile

```bash
make demo
```

The command deterministically regenerates `data/sample/incidents.csv` with seed 42 and writes the
run to `artifacts/demo/`. Demo metrics validate the software path only and are not research evidence.

## Corrected portfolio profile

Place a private Rekt-style JSON export or normalized CSV outside tracked Git, then run:

```bash
uv run python -m defi_security data build \
  --input /path/to/source-export.json \
  --output data/processed/incidents.csv
uv run python -m defi_security run --profile portfolio --robustness
```

The committed portfolio metrics were generated from 1,608 normalized incident records. The raw
records are excluded because redistribution rights have not been established.

## Conference profile

The 2025 paper metrics are stored in `artifacts/conference/reported_metrics.json`. To run the archival
configuration, provide the reconstructed paper dataset at `data/conference/incidents.csv` and use:

```bash
uv run python -m defi_security run --profile conference
```

This command never falls back to generated data.

## Neo4j

Neo4j is optional for the demo and model rerun. For graph loading:

```bash
docker compose up -d neo4j
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=local-password
uv run python -m defi_security graph load --input data/processed/incidents.csv
```

The loader uses `MERGE`; rerunning it does not duplicate protocol or incident nodes.

## Verification

```bash
make lint
make test
uv run python scripts/check_readme_metrics.py
make audit
```

Every run records the data SHA-256, library versions, timestamp, and Git commit in
`run_metadata.json`. Generated metadata is intentionally untracked because timestamps and local paths
are machine-specific.

