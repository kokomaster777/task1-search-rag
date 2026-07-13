"""RAG: вопрос -> эмбеддинг -> топ-k из Qdrant -> Claude формирует текстовый ответ."""
import os
from .vectors import vector_search

MODEL = os.environ.get("LLM_MODEL", "claude-haiku-4-5")

SYSTEM = """Ты — консультант интернет-магазина. Отвечай на вопрос покупателя,
опираясь ТОЛЬКО на переданные товары. Если подходящего товара нет — честно скажи об этом.
Указывай названия и цены. Отвечай кратко, по-русски."""


def answer(question: str, k: int = 5) -> dict:
    found = vector_search(question, limit=k)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {
            "question": question,
            "answer": None,
            "note": "ANTHROPIC_API_KEY не задан — возвращаю только результаты поиска",
            "retrieved": found,
        }
    import anthropic

    context = "\n".join(
        f"- {p['name']} | {p['category']} | {p['price']:.0f} руб. | {p['description']}" for p in found
    )
    resp = anthropic.Anthropic().messages.create(
        model=MODEL,
        max_tokens=500,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"Товары:\n{context}\n\nВопрос покупателя: {question}"}],
    )
    return {
        "question": question,
        "answer": resp.content[0].text,
        "retrieved": found,
        "usage": {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens},
    }
