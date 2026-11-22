
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"

QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME, device="cpu")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        embedding = self.model.encode(
            QUERY_INSTRUCTION + query,
            normalize_embeddings=True,
        )
        return embedding.tolist()
