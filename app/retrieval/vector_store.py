import atexit

from qdrant_client import QdrantClient, models


COLLECTION_NAME = "pdf_chunks"
VECTOR_SIZE = 384
TEXT_FIELD = "text"

client = QdrantClient(path="./data/qdrant")


def initialize_collection():
    """Ensure the collection and BM25 text index exist."""

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )

    collection_info = client.get_collection(COLLECTION_NAME)
    payload_schema = collection_info.payload_schema or {}

    if TEXT_FIELD not in payload_schema:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=TEXT_FIELD,
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                scoring=True,
            ),
        )


def store_chunks(chunks, embeddings):
    """Store document chunks and their dense embeddings."""

    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings count must match")

    if not chunks:
        raise ValueError("No chunks provided")

    points = [
        models.PointStruct(
            id=chunk["chunk_id"],
            vector=embedding,
            payload=chunk,
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )


def search_chunks(query_embedding, limit=5):
    """Retrieve chunks using dense vector search."""

    if limit <= 0:
        raise ValueError("Limit must be positive")

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=limit,
        with_payload=True,
    )

    return results.points

def get_document_chunks(document_id: str) -> list:
    """Retrieve all stored chunks belonging to a document."""
    results, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                )
            ]
        ),
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )

    return results

def close():
    """Close the Qdrant client."""

    client.close()


initialize_collection()
atexit.register(close)
