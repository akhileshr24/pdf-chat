
from uuid import NAMESPACE_URL, uuid5


CHUNK_SIZE = 400
CHUNK_OVERLAP = 60


def chunk_pages(pages: list[dict], document_id: str) -> list[dict]:
    """
    Split extracted PDF pages into overlapping word-based chunks.

    Each chunk gets a deterministic ID based on:
    document ID + page number + chunk index.
    """

    # Validate document ID
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("Document ID cannot be empty")

    # Validate chunk configuration
    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE must be positive")

    if CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError("CHUNK_OVERLAP must be between 0 and CHUNK_SIZE")

    chunks = []
    step_size = CHUNK_SIZE - CHUNK_OVERLAP

    for page in pages:
        page_number = page["page_number"]
        words = page["text"].split()

        # Skip pages without extractable text
        if not words:
            continue

        for chunk_index, start in enumerate(
            range(0, len(words), step_size)
        ):
            chunk_words = words[start:start + CHUNK_SIZE]

            # Generate a deterministic chunk ID
            chunk_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"hybrid-rag:{document_id}:{page_number}:{chunk_index}",
                )
            )

            chunks.append({
                "chunk_id": chunk_id,
                "document_id": document_id,
                "page_number": page_number,
                "chunk_index": chunk_index,
                "text": " ".join(chunk_words),
            })

    return chunks
