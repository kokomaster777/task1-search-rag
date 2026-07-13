"""Мини-бэкенд: сравнение полнотекстового и векторного поиска + RAG-ответы.

Эндпоинты:
  GET /search/fulltext?q=...   — Postgres FTS (русская морфология + триграммы)
  GET /search/vector?q=...     — Qdrant, семантический поиск
  GET /search/compare?q=...    — оба рядом, для наглядного сравнения
  POST /ask {"question": ...}  — RAG: retrieval + LLM, текстовый ответ
"""
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from . import db, vectors, rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open("/data/products.json", encoding="utf-8") as f:
        products = json.load(f)
    if db.init_and_ingest(products):
        print(f"[ingest] Postgres: загружено {len(products)} товаров")
    if vectors.init_and_ingest(products):
        print(f"[ingest] Qdrant: проиндексировано {len(products)} товаров")
    yield


app = FastAPI(title="Search comparison + RAG", lifespan=lifespan)


class Question(BaseModel):
    question: str
    k: int = 5


@app.get("/search/fulltext")
def search_fulltext(q: str, limit: int = 5):
    return {"engine": "postgres_fts", "query": q, "results": db.fulltext_search(q, limit)}


@app.get("/search/vector")
def search_vector(q: str, limit: int = 5):
    return {"engine": "qdrant_semantic", "query": q, "results": vectors.vector_search(q, limit)}


@app.get("/search/compare")
def search_compare(q: str, limit: int = 5):
    return {
        "query": q,
        "fulltext": db.fulltext_search(q, limit),
        "vector": vectors.vector_search(q, limit),
    }


@app.post("/ask")
def ask(body: Question):
    return rag.answer(body.question, body.k)


@app.get("/health")
def health():
    return {"status": "ok"}
