"""Embedding backends.

The in-memory store ships with a hashing embedder: deterministic, offline, and
adequate for lexical-ish matching over a tiny seeded corpus. It is *not* a
semantic embedding — two chunks that mean the same thing with different words
land nowhere near each other.

That is tolerable for the demo corpus and fatal for Pinecone: paying to store
hash vectors buys you a slower version of the in-memory store. So a real
embedder is required before the Pinecone backend is worth anything.

Three implementations behind one protocol:

- ``HashEmbedder``            offline default, no dependencies, not semantic
- ``OpenAICompatibleEmbedder`` any /embeddings endpoint (Nebius, OpenAI, ...)
- ``PineconeInferenceEmbedder`` Pinecone's hosted embedding models
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Protocol, runtime_checkable

from pathora.config import Settings, get_settings

HASH_DIM = 256


def tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+", text.lower())


@runtime_checkable
class Embedder(Protocol):
    name: str
    dimension: int

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else vector


class HashEmbedder:
    """Deterministic bag-of-words hashing. No network, no cost, no semantics."""

    name = "hash"

    def __init__(self, dimension: int = HASH_DIM) -> None:
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token, count in Counter(tokenize(text)).items():
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dimension
            vector[bucket] += float(count)
        return _normalize(vector)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class OpenAICompatibleEmbedder:
    """Any /embeddings endpoint: Nebius, OpenAI, Together, local vLLM.

    Reuses the same base-URL and key resolution as the chat provider, so one
    vendor and one key covers both generation and embedding.
    """

    name = "openai_compatible"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        from pathora.llm.providers import COMPATIBLE_ENDPOINTS

        vendor = self.settings.llm_provider.lower()
        default = COMPATIBLE_ENDPOINTS.get(vendor, COMPATIBLE_ENDPOINTS["openai"])
        self.base_url = (self.settings.embedding_base_url or default).rstrip("/")
        self.model = self.settings.embedding_model
        self.dimension = self.settings.embedding_dim
        self._local = any(h in self.base_url for h in ("localhost", "127.0.0.1"))
        if not self.settings.compatible_api_key and not self._local:
            raise RuntimeError("LLM_API_KEY (or OPENAI_API_KEY) is required for embeddings")

    async def _post(self, texts: list[str]) -> list[list[float]]:
        import httpx

        headers = {"content-type": "application/json"}
        if key := self.settings.compatible_api_key:
            headers["authorization"] = f"Bearer {key}"

        async with httpx.AsyncClient(timeout=self.settings.llm_timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model, "input": texts},
                headers=headers,
            )
            if response.status_code >= 400:
                raise RuntimeError(
                    f"{response.status_code} from {self.base_url}/embeddings: {response.text[:300]}"
                )
            body = response.json()

        # Order is not guaranteed to match input order; sort by index.
        rows = sorted(body["data"], key=lambda d: d.get("index", 0))
        return [row["embedding"] for row in rows]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # Batch to stay under request size limits on large corpora.
        batch = self.settings.embedding_batch_size
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch):
            vectors.extend(await self._post(texts[start : start + batch]))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        return (await self._post([text]))[0]


class PineconeInferenceEmbedder:  # pragma: no cover - requires live Pinecone
    """Pinecone's hosted embedding models, billed against Pinecone credits."""

    name = "pinecone"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        try:
            from pinecone import Pinecone
        except ImportError as exc:
            raise RuntimeError("pip install 'pathora-ai[infra]' to use Pinecone") from exc
        if not self.settings.pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY is not set")
        self._client = Pinecone(api_key=self.settings.pinecone_api_key)
        self.model = self.settings.embedding_model
        self.dimension = self.settings.embedding_dim

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        response = self._client.inference.embed(
            model=self.model,
            inputs=texts,
            parameters={"input_type": input_type, "truncate": "END"},
        )
        return [record["values"] for record in response.data]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import asyncio

        batch = self.settings.embedding_batch_size
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch):
            chunk = texts[start : start + batch]
            vectors.extend(await asyncio.to_thread(self._embed, chunk, "passage"))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        import asyncio

        return (await asyncio.to_thread(self._embed, [text], "query"))[0]


def build_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    match settings.embedding_backend.lower():
        case "openai" | "openai_compatible" | "nebius":
            return OpenAICompatibleEmbedder(settings)
        case "pinecone":
            return PineconeInferenceEmbedder(settings)
        case _:
            return HashEmbedder(settings.embedding_dim if settings.embedding_dim else HASH_DIM)
