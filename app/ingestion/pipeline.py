
from pathlib import Path

from app.ingestion.pdf_loader import extract_text
from app.ingestion.chunker import chunk_pages
from app.ingestion.file_hasher import calculate_file_hash
from app.retrieval.embedder import Embedder
from app.retrieval.vector_store import store_chunks, get_document_chunks


def ingest_pdf(
    pdf_path: str | Path,
    filename: str | None = None,
):
    # Keep the original filename instead of the temporary filename.
    filename = filename or Path(pdf_path).name

    # Step 1: Identify the PDF using its content.
    document_id = calculate_file_hash(pdf_path)

    # Step 2: Check whether this PDF was already ingested.
    existing_chunks = get_document_chunks(document_id)

    if existing_chunks:
        return {
            "document_id": document_id,
            "filename": existing_chunks[0].payload.get(
                "filename", filename
            ),
            "pages": len({
                chunk.payload["page_number"]
                for chunk in existing_chunks
            }),
            "chunks": len(existing_chunks),
            "already_exists": True,
        }

    # Step 3: Extract PDF text.
    pages = extract_text(pdf_path)

    # Step 4: Split text into chunks.
    chunks = chunk_pages(
        pages=pages,
        document_id=document_id,
    )

    # Step 5: Attach original filename to every chunk.
    for chunk in chunks:
        chunk["filename"] = filename

    # Step 6: Generate embeddings.
    embedder = Embedder()
    embeddings = embedder.embed_documents(
        [chunk["text"] for chunk in chunks]
    )

    # Step 7: Store chunks in Qdrant.
    store_chunks(chunks, embeddings)

    return {
        "document_id": document_id,
        "filename": filename,
        "pages": len(pages),
        "chunks": len(chunks),
        "already_exists": False,
    }
