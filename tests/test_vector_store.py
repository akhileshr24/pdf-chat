from uuid import uuid4
from app.retrieval.embedder import Embedder
from app.retrieval.vector_store import (
    store_chunks,
    search_chunks,
    close,
)


def test_store_and_search_chunks():
    embedder = Embedder()

    chunks = [
        {
            "chunk_id": str(uuid4()),
            "document_id": "test-doc-001",
            "page_number": 1,
            "text": "Python is a programming language.",
        }
    ]

    embeddings = embedder.embed_documents(
        [chunks[0]["text"]]
    )

    query_embedding = embedder.embed_query(
        "What is Python?"
    )

    store_chunks(chunks, embeddings)

    results = search_chunks(query_embedding, limit=1)

    assert len(results) == 1
    assert results[0].payload["text"] == chunks[0]["text"]

    close()
