"""Embedding backends.

The point of these tests is that the store must work with a *real* embedder,
not only the hash placeholder — and that hash vectors are never silently paired
with a paid vector database.
"""

from __future__ import annotations

import httpx
import pytest

from pathora.config import Settings
from pathora.rag.embeddings import (
    HashEmbedder,
    OpenAICompatibleEmbedder,
    build_embedder,
)
from pathora.rag.store import Document, InMemoryVectorStore


class TestHashEmbedder:
    async def test_deterministic(self):
        e = HashEmbedder()
        assert await e.embed_query("admissions") == await e.embed_query("admissions")

    async def test_unit_length(self):
        vector = await HashEmbedder().embed_query("admit rate")
        assert sum(v * v for v in vector) == pytest.approx(1.0, abs=1e-6)

    async def test_empty_text_does_not_crash(self):
        assert len(await HashEmbedder().embed_query("")) == 256

    async def test_is_not_semantic(self):
        """Documents the limitation that motivates a real embedder."""
        e = HashEmbedder()
        a = await e.embed_query("acceptance rate")
        b = await e.embed_query("admit rate")
        overlap = sum(x * y for x, y in zip(a, b, strict=True))
        assert overlap < 0.9, "hash embeddings should not capture synonymy"

    def test_is_the_default(self):
        assert isinstance(build_embedder(Settings()), HashEmbedder)


def stub_embeddings(dimension: int = 8, *, capture: dict | None = None):
    class _Response:
        status_code = 200
        text = ""

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    async def _post(self, url, json=None, headers=None):  # noqa: A002
        if capture is not None:
            capture["url"] = url
            capture["body"] = json
            capture["headers"] = headers
        rows = [
            {"index": i, "embedding": [float(i)] * dimension} for i in range(len(json["input"]))
        ]
        return _Response({"data": list(reversed(rows))})  # out of order on purpose

    return _post


class TestOpenAICompatibleEmbedder:
    @pytest.fixture
    def embedder(self):
        return OpenAICompatibleEmbedder(
            Settings(
                llm_provider="nebius",
                llm_api_key="k",
                embedding_backend="openai_compatible",
                embedding_model="BAAI/bge-multilingual-gemma2",
            )
        )

    def test_inherits_the_llm_vendor_endpoint(self, embedder):
        assert embedder.base_url == "https://api.studio.nebius.com/v1"

    async def test_hits_the_embeddings_route(self, embedder, monkeypatch):
        capture: dict = {}
        monkeypatch.setattr(httpx.AsyncClient, "post", stub_embeddings(capture=capture))
        await embedder.embed_query("test")
        assert capture["url"].endswith("/embeddings")
        assert capture["headers"]["authorization"] == "Bearer k"

    async def test_results_are_reordered_by_index(self, embedder, monkeypatch):
        """Providers may return embeddings out of order; misalignment is silent."""
        monkeypatch.setattr(httpx.AsyncClient, "post", stub_embeddings())
        vectors = await embedder.embed_documents(["a", "b", "c"])
        assert [v[0] for v in vectors] == [0.0, 1.0, 2.0]

    async def test_batches_large_inputs(self, monkeypatch):
        calls = []

        def counting_stub():
            inner = stub_embeddings()

            async def _post(self, url, json=None, headers=None):  # noqa: A002
                calls.append(len(json["input"]))
                return await inner(self, url, json=json, headers=headers)

            return _post

        embedder = OpenAICompatibleEmbedder(
            Settings(llm_provider="nebius", llm_api_key="k", embedding_batch_size=2)
        )
        monkeypatch.setattr(httpx.AsyncClient, "post", counting_stub())
        await embedder.embed_documents(["a", "b", "c", "d", "e"])
        assert calls == [2, 2, 1]

    async def test_empty_input_makes_no_request(self, embedder, monkeypatch):
        def explode(*a, **k):
            raise AssertionError("should not call the API for an empty batch")

        monkeypatch.setattr(httpx.AsyncClient, "post", explode)
        assert await embedder.embed_documents([]) == []

    def test_missing_key_fails_loudly(self):
        with pytest.raises(RuntimeError, match="LLM_API_KEY"):
            OpenAICompatibleEmbedder(
                Settings(llm_provider="nebius", embedding_backend="openai_compatible")
            )


class TestStoreUsesTheEmbedder:
    async def test_store_accepts_an_injected_embedder(self, settings, monkeypatch):
        embedder = OpenAICompatibleEmbedder(Settings(llm_provider="nebius", llm_api_key="k"))
        monkeypatch.setattr(httpx.AsyncClient, "post", stub_embeddings())
        store = InMemoryVectorStore(settings, embedder)
        await store.upsert(
            [
                Document(
                    id="d1",
                    university="U",
                    text="admissions",
                    source_url="https://u.example.edu",
                )
            ]
        )
        assert len(store) == 1
        assert await store.query("admissions", top_k=1)

    async def test_retrieval_still_works_with_the_hash_default(self, store):
        assert await store.query("admission requirements", top_k=3)
