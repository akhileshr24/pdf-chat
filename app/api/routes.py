
import logging
import tempfile
from pathlib import Path

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from pydantic import BaseModel

from app.ingestion.pipeline import ingest_pdf
from app.database.repository import (
    create_conversation,
    get_conversations,
    get_messages,
    get_conversation_documents,
    save_document,
    attach_document,
    save_message,
    update_conversation_title,
    conversation_exists,
)


logger = logging.getLogger(__name__)

router = APIRouter()

MAX_FILE_SIZE = 20 * 1024 * 1024

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"


class AskRequest(BaseModel):
    query: str
    conversation_id: str


def make_conversation_title(text, max_length=55):
    """Create a short readable title without using an LLM."""
    title = " ".join(text.split()).strip(" .?!")

    if not title:
        return "New conversation"

    if len(title) > max_length:
        title = title[:max_length - 3].rsplit(" ", 1)[0]
        title = title.rstrip(" .?!")
        title += "..."

    return title[0].upper() + title[1:]


# Conversation endpoints

@router.post("/conversations")
def new_conversation():
    conversation_id = create_conversation()

    return {"conversation_id": conversation_id}


@router.get("/conversations")
def list_conversations():
    return get_conversations()


@router.get("/conversations/{conversation_id}/messages")
def conversation_history(conversation_id: str):
    return get_messages(conversation_id)


@router.get("/conversations/{conversation_id}/documents")
def conversation_documents(conversation_id: str):
    return get_conversation_documents(conversation_id)


# PDF upload endpoint

@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    conversation_id: str = Form(...),
):
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed.",
        )

    # Validate conversation before processing the PDF.
    if not conversation_exists(conversation_id):
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    original_filename = Path(file.filename).name
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

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".pdf",
            prefix="upload_",
            dir=UPLOAD_DIR,
            delete=False,
        ) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        # Extract, chunk, embed, and store the PDF.
        result = ingest_pdf(
            pdf_path=temp_path,
            filename=original_filename,
        )

        document_id = result["document_id"]
        permanent_path = UPLOAD_DIR / f"{document_id}.pdf"

        # Keep only one permanent copy of each PDF.
        if permanent_path.exists():
            temp_path.unlink(missing_ok=True)
        else:
            temp_path.replace(permanent_path)

        # Save document metadata and conversation association.
        save_document(
            document_id=document_id,
            filename=original_filename,
            file_path=str(permanent_path),
        )

        attach_document(
            conversation_id,
            document_id,
        )

        # Use the PDF filename as the conversation title.
        update_conversation_title(
            conversation_id,
            make_conversation_title(Path(original_filename).stem),
        )

        message = (
            "PDF already exists and was added to this conversation."
            if result["already_exists"]
            else "PDF uploaded, processed, and saved successfully."
        )

        return {
            **result,
            "message": message,
        }

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception:
        logger.exception("PDF upload or processing failed")

        raise HTTPException(
            status_code=500,
            detail="PDF upload or processing failed. Check server logs.",
        )

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

        await file.close()


# Question-answer endpoint

@router.post("/ask")
def ask_question(payload: AskRequest, request: Request):
    query = payload.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    # Validate conversation before saving the question.
    if not conversation_exists(payload.conversation_id):
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    try:
        # Save the user's question.
        save_message(
            payload.conversation_id,
            "user",
            query,
        )

        # Generate an answer using the RAG pipeline.
        rag_pipeline = request.app.state.rag_pipeline

        result = rag_pipeline.ask(query)

        # Save the assistant's answer and citations.
        save_message(
            payload.conversation_id,
            "assistant",
            result["answer"],
            citations=result.get("citations", []),
            citation_verification=result.get(
                "citation_verification", {}
            ),
        )

        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception:
        logger.exception("Failed to generate an answer")

        raise HTTPException(
            status_code=500,
            detail="Failed to generate an answer. Check server logs.",
        )
