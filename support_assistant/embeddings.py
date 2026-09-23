import os
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DOCS_PATH = os.path.join(BASE_DIR, "docs")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "zepto_policies"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


def build_index():
    documents = []
    ids = []
    embeddings = []
    metadatas = []

    for filename in sorted(os.listdir(DOCS_PATH)):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(DOCS_PATH, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        document_id = filename.replace(".txt", "")

        embedding = model.encode(
            text,
            normalize_embeddings=True
        ).tolist()

        documents.append(text)
        ids.append(document_id)
        embeddings.append(embedding)
        metadatas.append({"source": filename})

    # Remove existing copies before rebuilding
    if ids:
        existing = collection.get(ids=ids)

        if existing["ids"]:
            collection.delete(ids=existing["ids"])

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"Indexed {len(ids)} documents.")
    print(f"Collection count: {collection.count()}")


def search(query, n_results=3):
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )


if __name__ == "__main__":
    build_index()

    test_query = "What is the delivery fee?"
    results = search(test_query, n_results=3)

    print("Test query:", test_query)
    print("Retrieved IDs:", results["ids"][0])
    print("Top result:")
    print(results["documents"][0][0][:300])
