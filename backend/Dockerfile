FROM python:3.12-slim

WORKDIR /srv

# CPU-версия PyTorch: ~200 МБ вместо ~6 ГБ с CUDA
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Прогреваем модель эмбеддингов в образ, чтобы старт был быстрым
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

COPY app ./app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]