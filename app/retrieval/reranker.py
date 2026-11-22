
from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RELEVANCE_THRESHOLD = -5.0


class Reranker:
    def __init__(self):
        self.model = CrossEncoder(
            MODEL_NAME,
            device="cpu",
        )

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Rerank chunks and reject low-relevance results."""

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if not results:
            return []

        sentence_pairs = [
            (query, result["payload"]["text"])
            for result in results
        ]

        scores = self.model.predict(
            sentence_pairs,
            show_progress_bar=False,
        )

        ranked_results = sorted(
            zip(results, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        relevant_results = [
            {
                "id": result["id"],
                "rrf_score": result["score"],
                "rerank_score": float(score),
                "payload": result["payload"],
            }
            for result, score in ranked_results
            if float(score) >= RELEVANCE_THRESHOLD
        ]

        return relevant_results[:top_k]
