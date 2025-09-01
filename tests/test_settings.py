from __future__ import annotations

import pytest

from defi_security.settings import Neo4jSettings


def test_neo4j_settings_require_secrets(monkeypatch) -> None:
    for name in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="Missing required Neo4j environment variables"):
        Neo4jSettings.from_env()


def test_neo4j_settings_use_consistent_names(monkeypatch) -> None:
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "test-password")
    settings = Neo4jSettings.from_env()
    assert settings.user == "neo4j"
    assert settings.database == "neo4j"

