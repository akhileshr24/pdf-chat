
from pathlib import Path

import pymupdf


def extract_text(pdf_path: str | Path) -> list[dict]:
    """Extract text from each PDF page."""

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise ValueError("Please provide a valid PDF file")

    pages = []

    try:
        with pymupdf.open(path) as document:
            if document.is_encrypted:
                raise ValueError("Encrypted PDFs are not supported")

            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()

                if text:
                    pages.append({
                        "page_number": page_number,
                        "text": text,
                    })

    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Could not read PDF: {path}") from exc

    if not pages:
        raise ValueError(
            "No extractable text found. This PDF may require OCR."
        )

    return pages
