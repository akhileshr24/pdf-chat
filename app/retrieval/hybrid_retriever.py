from app.retrieval.embedder import Embedder
from app.retrieval.vector_store import search_chunks
from app.retrieval.bm25_retriever import search_bm25


def fuse_results(
    dense_results: list,
    bm25_results: list,
    limit: int = 5,
    k: int = 60,
) -> list[dict]:
    """Combine dense and BM25 rankings using Reciprocal Rank Fusion."""

    if limit <= 0:
        raise ValueError("limit must be positive")

    if k <= 0:
        raise ValueError("k must be positive")

    scores = {}
    points = {}

    for results in [dense_results, bm25_results]:
        for rank, point in enumerate(results, start=1):
            point_id = point.id
            rrf_score = 1 / (k + rank)

            scores[point_id] = scores.get(point_id, 0) + rrf_score
            points[point_id] = point

    ranked_ids = sorted(
        scores,
        key=lambda point_id: scores[point_id],
        reverse=True,
    )

    return [
        {
            "id": point_id,
            "score": scores[point_id],
            "payload": points[point_id].payload,
        }
        for point_id in ranked_ids[:limit]
    ]


def hybrid_search(
    query: str,
    limit: int = 5,
    retrieval_limit: int = 10,
    embedder: Embedder | None = None,
) -> list[dict]:
    """Retrieve candidates using dense and BM25 search, then fuse them."""

    if not query.strip():
        raise ValueError("Query cannot be empty")

    if limit <= 0 or retrieval_limit <= 0:
        raise ValueError("Limits must be positive")

    embedder = embedder or Embedder()
    query_vector = embedder.embed_query(query)

    dense_results = search_chunks(
        query_vector,
        limit=retrieval_limit,
    )

    bm25_results = search_bm25(
        query,
        limit=retrieval_limit,
    )

    return fuse_results(
        dense_results,
        bm25_results,
        limit=limit,
    )
