from qdrant_client import models

from app.retrieval.vector_store import client, COLLECTION_NAME


def search_bm25(query: str, limit: int = 5):
    """Retrieve relevant chunks using BM25."""

    if not query.strip():
        raise ValueError("Query cannot be empty")

    if limit <= 0:
        raise ValueError("Limit must be positive")

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=models.TextQuery(text=query),
        using="text",
        limit=limit,
        with_payload=True,
    )

    return results.points
