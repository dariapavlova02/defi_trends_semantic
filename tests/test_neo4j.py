from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from defi_security.neo4j import Neo4jClient


@pytest.mark.asyncio
async def test_upsert_returns_processed_count() -> None:
    client = Neo4jClient(driver=object(), database="neo4j")
    client.write = AsyncMock(return_value=[{"processed": 2}])  # type: ignore[method-assign]
    count = await client.upsert_incidents(
        [
            {
                "event_id": "a",
                "incident_date": "2022-01-01T00:00:00Z",
                "loss_usd": 10.0,
                "incident_type": "oracle",
                "target_type": "dex",
                "protocol_id": "alpha",
            },
            {
                "event_id": "b",
                "incident_date": "2022-02-01T00:00:00Z",
                "loss_usd": 20.0,
                "incident_type": "access-control",
                "target_type": "bridge",
                "protocol_id": "beta",
            },
        ]
    )
    assert count == 2
    client.write.assert_awaited_once()

