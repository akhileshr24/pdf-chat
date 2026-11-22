
import pytest

from app.ingestion.pdf_loader import extract_text


def test_extract_text_returns_pages(tmp_path):
    import pymupdf

    pdf_path = tmp_path / "sample.pdf"

    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((50, 50), "Hello PDF")

        document.save(pdf_path)

    pages = extract_text(pdf_path)

    assert len(pages) == 1
    assert pages[0]["page_number"] == 1
    assert "Hello PDF" in pages[0]["text"]


def test_missing_pdf_raises_error(tmp_path):
    pdf_path = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        extract_text(pdf_path)


def test_non_pdf_file_raises_error(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Hello")

    with pytest.raises(ValueError):
        extract_text(file_path)


def test_empty_pdf_raises_error(tmp_path):
    import pymupdf

    pdf_path = tmp_path / "empty.pdf"

    with pymupdf.open() as document:
        document.new_page()
        document.save(pdf_path)

    with pytest.raises(ValueError):
        extract_text(pdf_path)
