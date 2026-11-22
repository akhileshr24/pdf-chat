
from app.retrieval.embedder import Embedder


def test_query_embedding_dimension():
    embedder = Embedder()

    vector = embedder.embed_query("What is machine learning?")

    assert len(vector) == 384


def test_document_embedding_dimension():
    embedder = Embedder()

    vectors = embedder.embed_documents([
        "Machine learning is a branch of artificial intelligence."
    ])

    assert len(vectors) == 1
    assert len(vectors[0]) == 384
