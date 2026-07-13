"""Векторный (семантический) поиск: multilingual-e5-small + Qdrant.
Важно для e5: документы кодируются с префиксом "passage: ", запросы — "query: "."""
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

COLLECTION = "products"
_model: SentenceTransformer | None = None
_client: QdrantClient | None = None


def model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("intfloat/multilingual-e5-small")
    return _model


def client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    return _client


def init_and_ingest(products: list[dict]) -> bool:
    c = client()
    if c.collection_exists(COLLECTION):
        if c.count(COLLECTION).count >= len(products):
            return False
    else:
        c.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=model().get_sentence_embedding_dimension(), distance=Distance.COSINE),
        )
    texts = [f"passage: {p['name']}. {p['category']}. {p['description']}" for p in products]
    vecs = model().encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    c.upsert(
        COLLECTION,
        points=[PointStruct(id=p["id"], vector=v.tolist(), payload=p) for p, v in zip(products, vecs)],
    )
    return True


def vector_search(query: str, limit: int = 5) -> list[dict]:
    vec = model().encode(f"query: {query}", normalize_embeddings=True).tolist()
    hits = client().query_points(COLLECTION, query=vec, limit=limit).points
    return [{**h.payload, "score": round(h.score, 4)} for h in hits]
