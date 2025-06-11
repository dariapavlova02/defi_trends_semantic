from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from defi_security.settings import Neo4jSettings


class Neo4jClient:
    def __init__(self, driver: Any, database: str):
        self._driver = driver
        self._database = database

    @classmethod
    async def create(cls, settings: Neo4jSettings) -> Neo4jClient:
        try:
            from neo4j import AsyncGraphDatabase
        except ImportError as exc:
            raise RuntimeError("Install the graph extra: pip install -e '.[graph]'") from exc
        driver = AsyncGraphDatabase.driver(
            settings.uri,
            auth=(settings.user, settings.password),
            connection_timeout=10,
        )
        await driver.verify_connectivity()
        return cls(driver, settings.database)

    async def __aenter__(self) -> Neo4jClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._driver.close()

    async def read(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, parameters or {})
            return [record.data() async for record in result]

    async def write(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, parameters or {})
            records = [record.data() async for record in result]
            await result.consume()
            return records

    async def upsert_incidents(self, rows: Iterable[dict[str, Any]]) -> int:
        payload = list(rows)
        query = """
        UNWIND $rows AS row
        MERGE (p:Protocol {id: row.protocol_id})
        MERGE (e:SecurityIncident {id: row.event_id})
        SET e.incident_date = datetime(row.incident_date),
            e.loss_usd = row.loss_usd,
            e.incident_type = row.incident_type,
            e.target_type = row.target_type
        MERGE (p)-[:HAS_INCIDENT]->(e)
        RETURN count(e) AS processed
        """
        result = await self.write(query, {"rows": payload})
        return int(result[0]["processed"]) if result else 0

