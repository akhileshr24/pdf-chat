
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.generation.pipeline import RAGPipeline
from app.database.connection import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize SQLite database
    init_db()

    # Initialize the RAG pipeline once at startup
    app.state.rag_pipeline = RAGPipeline()

    yield

    # Application shutdown
    del app.state.rag_pipeline


app = FastAPI(
    title="Hybrid RAG API",
    description="PDF-based Hybrid Retrieval-Augmented Generation",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health")
def health_check():
    return {"status": "healthy"}
