"""Checkpointer selection (Section 23).

Human-in-the-loop resume depends entirely on the checkpointer: an interrupt is
only resumable if the state it paused on survives. ``MemorySaver`` does not
survive a process restart, so a single-container deployment that reboots loses
every in-flight analysis. ``CHECKPOINT_BACKEND=postgres`` is the production
setting.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from pathora.config import Settings, get_settings


def build_checkpointer(settings: Settings | None = None) -> Any:
    settings = settings or get_settings()
    backend = settings.checkpoint_backend.lower()

    if backend == "memory":
        return MemorySaver()

    if backend == "postgres":  # pragma: no cover - requires a live Postgres
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
        except ImportError as exc:
            raise RuntimeError(
                "CHECKPOINT_BACKEND=postgres requires langgraph-checkpoint-postgres: "
                "pip install 'pathora-ai[infra]'"
            ) from exc

        dsn = settings.database_url.replace("postgresql+psycopg://", "postgresql://")
        if not dsn.startswith("postgresql://"):
            raise RuntimeError(
                "CHECKPOINT_BACKEND=postgres requires a PostgreSQL DATABASE_URL, "
                f"got {settings.database_url!r}"
            )
        saver = PostgresSaver.from_conn_string(dsn)
        saver.setup()
        return saver

    raise ValueError(
        f"unknown CHECKPOINT_BACKEND {settings.checkpoint_backend!r} (expected memory|postgres)"
    )
