
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from app.ingestion.pipeline import ingest_pdf


router = APIRouter()

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


class AskRequest(BaseModel):
    query: str


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed.",
        )

    original_filename = file.filename
    temp_path = None

    try:
        content = await file.read(MAX_FILE_SIZE + 1)

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File size exceeds the 20 MB limit.",
            )

        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        result = ingest_pdf(
            pdf_path=temp_path,
            filename=original_filename,
        )

        return {
            "message": (
                "PDF already exists."
                if result["already_exists"]
                else "PDF uploaded and processed successfully."
            ),
            **result,
        }

    except HTTPException:
        raise

    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="PDF processing failed.",
        )

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

        await file.close()


@router.post("/ask")
def ask_question(payload: AskRequest, request: Request):
    query = payload.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        rag_pipeline = request.app.state.rag_pipeline
        result = rag_pipeline.ask(query)
        return result

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate an answer.",
        )
