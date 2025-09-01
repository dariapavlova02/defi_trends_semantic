from __future__ import annotations

import os

import pytest

from defi_security.neo4j import Neo4jClient
from defi_security.settings import Neo4jSettings

pytestmark = pytest.mark.neo4j


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("RUN_NEO4J_INTEGRATION") != "1",
    reason="set RUN_NEO4J_INTEGRATION=1 to test a running Neo4j service",
)
async def test_incident_upsert_is_idempotent() -> None:
    row = {
        "event_id": "ci-integration-event",
        "incident_date": "2024-01-01T00:00:00Z",
        "loss_usd": 100.0,
        "incident_type": "test",
        "target_type": "test",
        "protocol_id": "ci-integration-protocol",
    }
    client = await Neo4jClient.create(Neo4jSettings.from_env())
    async with client:
        assert await client.upsert_incidents([row]) == 1
        assert await client.upsert_incidents([row]) == 1
        result = await client.read(
            "MATCH (:Protocol {id: $protocol})-[:HAS_INCIDENT]->(e:SecurityIncident {id: $event}) "
            "RETURN count(e) AS count",
            {"protocol": row["protocol_id"], "event": row["event_id"]},
        )
    assert result == [{"count": 1}]

