"""Retrieval layer (Section 19).

Two backends behind one Protocol:
- ``InMemoryVectorStore``  — hash embeddings + lexical scoring, zero infra, used
  by tests, CI and degraded mode.
- ``PineconeVectorStore``  — same interface against a Pinecone index.

Retrieval is hybrid: a dense score (embedding cosine) is blended with a sparse
lexical score, then filtered by university metadata. Reranking (by source
authority, per Section 17) is optional and configurable.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from pathora.config import Settings, get_settings
from pathora.domain.models import SOURCE_PRIORITY, SourceType

EMBED_DIM = 256
TOKEN_RE = re.compile(r"[a-z0-9]+")


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    university: str
    title: str = ""
    text: str
    source_url: str
    source_type: SourceType = "other"
    published_at: datetime | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    depth: int = 0
    facts: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(Document):
    score: float = 0.0


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def embed(text: str, dim: int = EMBED_DIM) -> list[float]:
    """Deterministic hashing embedding. Swap for a real embedding model in prod."""
    vector = [0.0] * dim
    for token, count in Counter(tokenize(text)).items():
        bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % dim
        vector[bucket] += float(count)
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else vector


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


class VectorStore(Protocol):
    async def upsert(self, documents: list[Document]) -> int: ...

    async def query(
        self,
        query: str,
        *,
        university: str | None = None,
        top_k: int = 6,
        max_depth: int = 0,
        rerank: bool | None = None,
    ) -> list[RetrievedChunk]: ...


class InMemoryVectorStore:
    backend = "memory"

    def __init__(self, settings: Settings | None = None, embedder: Any = None) -> None:
        self.settings = settings or get_settings()
        from pathora.rag.embeddings import build_embedder

        self.embedder = embedder or build_embedder(self.settings)
        self._docs: dict[str, Document] = {}
        self._vectors: dict[str, list[float]] = {}

    async def upsert(self, documents: list[Document]) -> int:
        if not documents:
            return 0
        vectors = await self.embedder.embed_documents([f"{d.title} {d.text}" for d in documents])
        for doc, vector in zip(documents, vectors, strict=True):
            self._docs[doc.id] = doc
            self._vectors[doc.id] = vector
        return len(documents)

    async def query(
        self,
        query: str,
        *,
        university: str | None = None,
        top_k: int = 6,
        max_depth: int = 0,
        rerank: bool | None = None,
    ) -> list[RetrievedChunk]:
        query_vector = await self.embedder.embed_query(query)
        query_tokens = set(tokenize(query))
        results: list[RetrievedChunk] = []

        for doc_id, doc in self._docs.items():
            if university and doc.university.lower() != university.lower():
                continue
            if doc.depth > max_depth:
                continue

            dense = cosine(query_vector, self._vectors[doc_id])
            doc_tokens = set(tokenize(f"{doc.title} {doc.text}"))
            sparse = len(query_tokens & doc_tokens) / len(query_tokens or {"x"})
            score = 0.6 * dense + 0.4 * sparse
            results.append(RetrievedChunk(**doc.model_dump(), score=round(score, 6)))

        use_rerank = self.settings.rag_rerank_enabled if rerank is None else rerank
        if use_rerank:
            results.sort(key=lambda c: (SOURCE_PRIORITY.get(c.source_type, 9), -c.score, c.id))
        else:
            results.sort(key=lambda c: (-c.score, c.id))
        return results[:top_k]

    def __len__(self) -> int:
        return len(self._docs)

    def __bool__(self) -> bool:
        # Without this, an empty store is falsy and `store or build_store()`
        # style defaulting silently swaps it for a different instance.
        return True


class PineconeVectorStore:  # pragma: no cover - requires live Pinecone
    backend = "pinecone"

    def __init__(self, settings: Settings | None = None, embedder: Any = None) -> None:
        self.settings = settings or get_settings()
        from pathora.rag.embeddings import build_embedder

        self.embedder = embedder or build_embedder(self.settings)
        try:
            from pinecone import Pinecone  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pip install 'pathora-ai[infra]' to use Pinecone") from exc
        if not self.settings.pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY is not set")
        self._index = Pinecone(api_key=self.settings.pinecone_api_key).Index(
            self.settings.pinecone_index
        )

    async def upsert(self, documents: list[Document]) -> int:
        embeddings = await self.embedder.embed_documents([f"{d.title} {d.text}" for d in documents])
        vectors = [
            {
                "id": doc.id,
                "values": vector,
                "metadata": {
                    "university": doc.university,
                    "title": doc.title,
                    "text": doc.text,
                    "source_url": doc.source_url,
                    "source_type": doc.source_type,
                    "published_at": doc.published_at.isoformat() if doc.published_at else None,
                    "retrieved_at": doc.retrieved_at.isoformat(),
                    "depth": doc.depth,
                    "facts": json.dumps(doc.facts),
                },
            }
            for doc, vector in zip(documents, embeddings, strict=True)
        ]
        self._index.upsert(vectors=vectors)
        return len(vectors)

    async def query(
        self,
        query: str,
        *,
        university: str | None = None,
        top_k: int = 6,
        max_depth: int = 0,
        rerank: bool | None = None,
    ) -> list[RetrievedChunk]:
        flt: dict[str, Any] = {"depth": {"$lte": max_depth}}
        if university:
            flt["university"] = {"$eq": university}
        response = self._index.query(
            vector=await self.embedder.embed_query(query),
            top_k=top_k,
            include_metadata=True,
            filter=flt,
        )
        chunks = [
            RetrievedChunk(
                id=match["id"],
                university=match["metadata"]["university"],
                title=match["metadata"].get("title", ""),
                text=match["metadata"].get("text", ""),
                source_url=match["metadata"]["source_url"],
                source_type=match["metadata"].get("source_type", "other"),
                published_at=match["metadata"].get("published_at"),
                depth=int(match["metadata"].get("depth", 0)),
                facts=json.loads(match["metadata"].get("facts", "{}")),
                score=match.get("score", 0.0),
            )
            for match in response.get("matches", [])
        ]
        use_rerank = self.settings.rag_rerank_enabled if rerank is None else rerank
        if use_rerank:
            chunks.sort(key=lambda c: (SOURCE_PRIORITY.get(c.source_type, 9), -c.score, c.id))
        return chunks


def build_store(settings: Settings | None = None, embedder: Any = None) -> VectorStore:
    settings = settings or get_settings()
    if settings.vector_backend.lower() == "pinecone":
        return PineconeVectorStore(settings, embedder)
    return InMemoryVectorStore(settings, embedder)


SEED_DIR = Path(__file__).resolve().parents[3] / "data/seed"
SEED_PATH = SEED_DIR / "colleges.json"


def corpus_files(directory: Path | None = None) -> list[Path]:
    """Every corpus file the app should load.

    ``colleges.json`` (the synthetic demo corpus) plus any ``*.colleges.json``
    dropped alongside it. Ingesting real universities writes such a file, so a
    restart picks them up without editing code.
    """
    directory = directory or SEED_DIR
    files = [SEED_PATH] if SEED_PATH.exists() else []
    files += sorted(p for p in directory.glob("*.colleges.json"))
    return files


def load_seed_payload(path: Path | None = None) -> dict[str, Any]:
    """Load and merge corpus files. Later files win on id/university collisions."""
    paths = [path] if path is not None else corpus_files()

    colleges: dict[str, dict[str, Any]] = {}
    documents: dict[str, dict[str, Any]] = {}
    for file in paths:
        payload = json.loads(file.read_text())
        for college in payload.get("colleges", []):
            colleges[college["university"]] = college
        for doc in payload.get("documents", []):
            documents[doc["id"]] = doc

    return {"colleges": list(colleges.values()), "documents": list(documents.values())}


async def seeded_store(settings: Settings | None = None, path: Path | None = None):
    """Return an in-memory store preloaded with every available corpus file."""
    payload = load_seed_payload(path)
    store = InMemoryVectorStore(settings)
    await store.upsert([Document(**doc) for doc in payload["documents"]])
    return store
