FROM python:3.11-slim

WORKDIR /app

COPY support_assistant/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

# Download the required embedding model during image build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy support assistant source and corpus
COPY support_assistant /app/support_assistant

# Build the ChromaDB index inside the image
RUN python -m support_assistant.embeddings

EXPOSE 7860

CMD ["uvicorn", "support_assistant.main:app", "--host", "0.0.0.0", "--port", "7860"]
