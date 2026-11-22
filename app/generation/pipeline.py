
from app.retrieval.embedder import Embedder
from app.retrieval.hybrid_retriever import hybrid_search
from app.retrieval.reranker import Reranker
from app.generation.llm import LLMGenerator


class RAGPipeline:
    def __init__(self):
        """Initialize reusable RAG components."""

        self.embedder = Embedder()
        self.reranker = Reranker()
        self.generator = LLMGenerator()

    def ask(
        self,
        query: str,
        retrieval_limit: int = 10,
        context_limit: int = 3,
    ) -> dict:
        """Retrieve, filter, rerank, and generate an answer."""

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if retrieval_limit <= 0 or context_limit <= 0:
            raise ValueError("Limits must be positive")

        # Step 1: Hybrid retrieval
        candidates = hybrid_search(
            query=query,
            limit=retrieval_limit,
            retrieval_limit=retrieval_limit,
            embedder=self.embedder,
        )

        if not candidates:
            return {
                "answer": "I don't know based on the provided PDF context.",
                "citations": [],
                "citation_verification": {
                    "status": "NOT_REQUIRED",
                    "claims": [],
                },
            }

        # Step 2: Cross-Encoder reranking + relevance filtering
        reranked_chunks = self.reranker.rerank(
            query=query,
            results=candidates,
            top_k=context_limit,
        )

        # Step 3: Abstain if no relevant chunks remain
        if not reranked_chunks:
            return {
                "answer": "I don't know based on the provided PDF context.",
                "citations": [],
                "citation_verification": {
                    "status": "NOT_REQUIRED",
                    "claims": [],
                },
            }

        # Step 4: Generate answer using Groq
        return self.generator.generate_answer(
            query=query,
            context_chunks=reranked_chunks,
        )
