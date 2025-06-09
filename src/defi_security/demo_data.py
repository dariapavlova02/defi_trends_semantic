from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd


def generate_demo_data(rows: int = 180, seed: int = 42) -> pd.DataFrame:
    """Create an explicitly synthetic, deterministic integration-test dataset."""
    if rows < 60:
        raise ValueError("Demo data requires at least 60 rows")
    rng = np.random.default_rng(seed)
    start = datetime(2021, 1, 3, tzinfo=UTC)
    records = []
    incident_types = ["oracle", "access-control", "reentrancy", "economic"]
    target_types = ["dex", "lending", "bridge"]
    chains = ["ethereum", "arbitrum", "polygon", "bsc"]
    seen: dict[str, int] = {}
    for index in range(rows):
        protocol_number = index % 18
        protocol_id = f"protocol-{protocol_number:02d}"
        past_count = seen.get(protocol_id, 0)
        seen[protocol_id] = past_count + 1
        incident_date = start + timedelta(days=index * 9)
        chain_count = 1 + protocol_number % 4
        is_fork = int(protocol_number % 3 == 0)
        child_count = protocol_number % 5 if is_fork else 0
        risk_signal = 0.38 * past_count + 0.42 * chain_count + 0.55 * is_fork + 0.15 * child_count
        loss_usd = float(np.exp(11.7 + risk_signal + rng.normal(0, 0.85)))
        available_at = incident_date - timedelta(days=1)
        records.append(
            {
                "event_id": f"demo-{index:04d}",
                "incident_date": incident_date.isoformat(),
                "loss_usd": round(loss_usd, 2),
                "incident_type": incident_types[index % len(incident_types)],
                "target_type": target_types[protocol_number % len(target_types)],
                "chain": chains[protocol_number % len(chains)],
                "protocol_id": protocol_id,
                "protocol_chains_count": chain_count,
                "protocol_chains_count_available_at": available_at.isoformat(),
                "is_forked_from_parent": is_fork,
                "is_forked_from_parent_available_at": available_at.isoformat(),
                "parent_fork_children_count": child_count,
                "parent_fork_children_count_available_at": available_at.isoformat(),
                "protocol_past_events_count": past_count,
                "protocol_past_events_count_available_at": available_at.isoformat(),
            }
        )
    return pd.DataFrame(records)

