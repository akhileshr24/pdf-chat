from app.ingestion.pipeline import ingest_pdf


def test_ingest_pdf():
    result = ingest_pdf("data/pdfs/RAG_Research_Paper.pdf")

    assert result["document_id"]
    assert result["pages"] > 0
    assert result["chunks"] > 0

    print(result)