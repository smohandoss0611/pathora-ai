"""Create the Pinecone index and load the corpus into it.

    python scripts/setup_pinecone.py --dry-run   # show what would happen
    python scripts/setup_pinecone.py             # create index + ingest

The index dimension must match the embedding model exactly, and Pinecone will
not let you change it afterwards — you would have to delete and recreate. This
script derives the dimension from the configured embedder by embedding one
probe string, so a mismatch is caught before anything is written.

It also refuses to run with EMBEDDING_BACKEND=hash, because hash vectors are not
semantic: storing them in a paid vector database gives you a slower, costlier
version of the in-memory store.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathora.config import Settings  # noqa: E402
from pathora.rag.embeddings import build_embedder  # noqa: E402
from pathora.rag.ingest import to_documents  # noqa: E402
from pathora.rag.store import PineconeVectorStore, load_seed_payload  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--metric", default="cosine", choices=["cosine", "dotproduct", "euclidean"])
    parser.add_argument("--cloud", default="aws")
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    settings = Settings()
    print(f"index            : {settings.pinecone_index}")
    print(f"embedding backend: {settings.embedding_backend}")
    print(f"embedding model  : {settings.embedding_model}")

    if settings.embedding_backend.lower() == "hash":
        print(
            "\nRefusing to continue: EMBEDDING_BACKEND=hash produces non-semantic\n"
            "vectors. Set EMBEDDING_BACKEND=openai_compatible (uses your LLM key)\n"
            "or pinecone (uses Pinecone hosted inference) first."
        )
        return 1

    if not settings.pinecone_api_key:
        print("\nPINECONE_API_KEY is not set.")
        return 1

    # Probe the real dimension rather than trusting the configured value.
    embedder = build_embedder(settings)
    probe = await embedder.embed_query("dimension probe")
    dimension = len(probe)
    print(f"measured dim     : {dimension}")
    if dimension != settings.embedding_dim:
        print(
            f"\nNote: EMBEDDING_DIM is {settings.embedding_dim} but the model returns "
            f"{dimension}. Using {dimension}; update your .env to match."
        )

    payload = load_seed_payload()
    documents = [doc for record in payload["documents"] for doc in to_documents(record)]
    print(f"documents        : {len(payload['documents'])} -> {len(documents)} chunks")

    if args.dry_run:
        print("\n--dry-run: nothing created, nothing uploaded.")
        return 0

    from pinecone import Pinecone, ServerlessSpec

    client = Pinecone(api_key=settings.pinecone_api_key)
    existing = {i["name"] for i in client.list_indexes()}

    if settings.pinecone_index in existing:
        current = client.describe_index(settings.pinecone_index)["dimension"]
        if current != dimension:
            print(
                f"\nIndex '{settings.pinecone_index}' exists with dimension {current}, "
                f"but the embedder produces {dimension}. Pinecone cannot change an "
                f"index dimension — delete it and re-run, or use a different "
                f"PINECONE_INDEX name."
            )
            return 1
        print(f"index exists with matching dimension {current}")
    else:
        print(f"creating index (dim={dimension}, metric={args.metric})...")
        client.create_index(
            name=settings.pinecone_index,
            dimension=dimension,
            metric=args.metric,
            spec=ServerlessSpec(cloud=args.cloud, region=args.region),
        )

    store = PineconeVectorStore(settings, embedder)
    count = await store.upsert(documents)
    print(f"\nUpserted {count} chunks into '{settings.pinecone_index}'.")
    print("Set VECTOR_BACKEND=pinecone and restart the app to use it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
