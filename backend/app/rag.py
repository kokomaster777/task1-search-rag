"""RAG через локальную Ollama: вопрос -> эмбеддинг -> топ-k из Qdrant -> LLM -> ответ.
Ключ не нужен: модель работает на твоём компьютере (Ollama слушает порт 11434).
Бэкенд живёт в docker, поэтому наружу к Ollama ходим через host.docker.internal.

ЗАМЕНЯЕТ файл backend/app/rag.py"""
import os
import requests

from .vectors import vector_search

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434")
MODEL = os.environ.get("LLM_MODEL", "qwen2.5:1.5b-instruct")

SYSTEM = """Ты — консультант интернет-магазина. Отвечай на вопрос покупателя,
опираясь ТОЛЬКО на переданные товары. Если подходящего товара нет — честно скажи об этом.
Указывай названия и цены. Отвечай кратко, по-русски."""


def answer(question: str, k: int = 5) -> dict:
    found = vector_search(question, limit=k)
    context = "\n".join(
        f"- {p['name']} | {p['category']} | {p['price']:.0f} руб. | {p['description']}" for p in found
    )
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 400},
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Товары:\n{context}\n\nВопрос покупателя: {question}"},
                ],
            },
            timeout=300,  # маленькая модель на CPU может думать долго
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "question": question,
            "answer": data["message"]["content"],
            "retrieved": found,
            "usage": {
                "input_tokens": data.get("prompt_eval_count", 0),
                "output_tokens": data.get("eval_count", 0),
                "model": MODEL,
            },
        }
    except requests.exceptions.ConnectionError:
        return {
            "question": question,
            "answer": None,
            "note": f"Ollama недоступна по адресу {OLLAMA_URL}. Запусти Ollama на хосте "
                    f"и убедись, что модель скачана: ollama pull {MODEL}",
            "retrieved": found,
        }
