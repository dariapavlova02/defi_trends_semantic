from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Neo4jSettings:
    uri: str
    user: str
    password: str
    database: str = "neo4j"

    @classmethod
    def from_env(cls) -> Neo4jSettings:
        values = {
            "uri": os.getenv("NEO4J_URI"),
            "user": os.getenv("NEO4J_USER"),
            "password": os.getenv("NEO4J_PASSWORD"),
            "database": os.getenv("NEO4J_DATABASE", "neo4j"),
        }
        missing = [key for key in ("uri", "user", "password") if not values[key]]
        if missing:
            names = ", ".join(f"NEO4J_{name.upper()}" for name in missing)
            raise ValueError(f"Missing required Neo4j environment variables: {names}")
        return cls(**values)  # type: ignore[arg-type]

