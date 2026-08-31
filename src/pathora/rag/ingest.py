"""Source -> clean -> chunk -> embed -> store (Section 19).

Every chunk carries university / source_url / source_type / published_at /
retrieved_at metadata so the Evidence Passport can be reconstructed later.
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pathora.rag.store import Document, VectorStore, build_store, load_seed_payload

WHITESPACE = re.compile(r"[ \t\u00a0]+")
BLANK_LINES = re.compile(r"\n{3,}")
BOILERPLATE = re.compile(
    r"^(skip to main content|cookie policy|©.*|share this page|print this page)$",
    re.I | re.M,
)


def clean(text: str) -> str:
    text = BOILERPLATE.sub("", text)
    text = WHITESPACE.sub(" ", text)
    text = BLANK_LINES.sub("\n\n", text)
    return text.strip()


def chunk(text: str, *, size: int = 900, overlap: int = 120) -> list[str]:
    """Sentence-aware chunking with character overlap."""
    if size <= 0:
        raise ValueError("size must be positive")
    sentences = re.split(r"(?<=[.!?])\s+", clean(text))
    chunks: list[str] = []
    buffer = ""

    for sentence in sentences:
        if not sentence:
            continue
        if len(buffer) + len(sentence) + 1 <= size:
            buffer = f"{buffer} {sentence}".strip()
            continue
        if buffer:
            chunks.append(buffer)
        buffer = (buffer[-overlap:] + " " + sentence).strip() if overlap and buffer else sentence
        while len(buffer) > size:
            chunks.append(buffer[:size])
            buffer = buffer[size - overlap :]

    if buffer:
        chunks.append(buffer)
    return chunks


def to_documents(record: dict[str, Any]) -> list[Document]:
    """Expand one source record into chunk-level documents."""
    pieces = chunk(record["text"])
    now = datetime.now(UTC)
    return [
        Document(
            id=record["id"] if len(pieces) == 1 else f"{record['id']}#{index}",
            university=record["university"],
            title=record.get("title", ""),
            text=piece,
            source_url=record["source_url"],
            source_type=record.get("source_type", "other"),
            published_at=record.get("published_at"),
            retrieved_at=record.get("retrieved_at", now),
            depth=record.get("depth", 0),
            facts=record.get("facts", {}),
        )
        for index, piece in enumerate(pieces)
    ]


async def ingest_records(records: list[dict[str, Any]], store: VectorStore | None = None) -> int:
    # `store or build_store()` is a trap here: InMemoryVectorStore defines
    # __len__, so a freshly created (empty) store is falsy and the caller's
    # store would be silently discarded in favour of a throwaway.
    if store is None:
        store = build_store()
    documents = [doc for record in records for doc in to_documents(record)]
    return await store.upsert(documents)


async def ingest_seed(store: VectorStore | None = None, path: Path | None = None) -> int:
    return await ingest_records(load_seed_payload(path)["documents"], store)


def main() -> None:  # pragma: no cover - CLI entry point
    count = asyncio.run(ingest_seed())
    print(f"ingested {count} chunks into {build_store().__class__.__name__}")


if __name__ == "__main__":  # pragma: no cover
    main()
