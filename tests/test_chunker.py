import pytest

from app.ingestion.chunker import chunk_pages


def test_chunk_pages_returns_chunks():
    pages = [
        {
            "page_number": 1,
            "text": " ".join(["hello"] * 500),
        }
    ]

    chunks = chunk_pages(pages, "doc-001")

    assert len(chunks) == 2
    assert chunks[0]["document_id"] == "doc-001"
    assert chunks[0]["page_number"] == 1
    assert len(chunks[0]["text"].split()) == 400


def test_chunk_overlap():
    pages = [
        {
            "page_number": 1,
            "text": " ".join([f"word{i}" for i in range(500)]),
        }
    ]

    chunks = chunk_pages(pages, "doc-001")

    first_words = chunks[0]["text"].split()
    second_words = chunks[1]["text"].split()

    assert first_words[-60:] == second_words[:60]


def test_empty_document_id_raises_error():
    with pytest.raises(ValueError):
        chunk_pages([{"page_number": 1, "text": "hello"}], "")


def test_empty_page_is_skipped():
    pages = [{"page_number": 1, "text": "   "}]

    chunks = chunk_pages(pages, "doc-001")

    assert chunks == []


def test_chunk_ids_are_deterministic():
    pages = [
        {
            "page_number": 1,
            "text": " ".join([f"word{i}" for i in range(500)]),
        }
    ]

    first_run = chunk_pages(pages, "doc-001")
    second_run = chunk_pages(pages, "doc-001")

    first_ids = [chunk["chunk_id"] for chunk in first_run]
    second_ids = [chunk["chunk_id"] for chunk in second_run]

    assert first_ids == second_ids
    assert len(first_ids) == len(set(first_ids))


def test_different_documents_have_different_chunk_ids():
    pages = [{"page_number": 1, "text": "hello world"}]

    first = chunk_pages(pages, "doc-001")
    second = chunk_pages(pages, "doc-002")

    assert first[0]["chunk_id"] != second[0]["chunk_id"]
