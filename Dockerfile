FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 HF_HOME=/app/.hf

# CPU-only torch first, so sentence-transformers doesn't pull the multi-GB CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -c "import nltk; nltk.download('punkt_tab')"

# bake the models into the image so containers start without downloading them
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
CrossEncoder('cross-encoder/quora-distilroberta-base')"

COPY rag rag
COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
