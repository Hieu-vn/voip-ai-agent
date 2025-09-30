import httpx
import json
import os
from qdrant_client import QdrantClient, models

# Configuration
TEI_URL = os.getenv("TEI_URL", "http://localhost:8080")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "./data/knowledge_base.json")
COLLECTION_NAME = "kb_vi"
VECTOR_SIZE = 1024 # BGE-M3 embedding size

def get_embeddings(texts: list[str]) -> list[list[float]]:
    r = httpx.post(f"{TEI_URL}/embed", json={"inputs": texts})
    r.raise_for_status()
    return r.json()["embeddings"]

def ingest_data():
    client = QdrantClient(host=QDRANT_URL.split("//")[-1].split(":")[0], port=int(QDRANT_URL.split(":")[-1]))

    # Create collection if it doesn't exist
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
    )

    # Load sample data
    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    points = []
    for i, item in enumerate(data):
        text = item.get("text", "")
        if not text:
            continue
        
        # Get embedding from TEI
        embedding = get_embeddings([text])[0]
        
        points.append(
            models.PointStruct(
                id=i,
                vector=embedding,
                payload=item,
            )
        )
    
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Ingested {len(points)} points into Qdrant collection {COLLECTION_NAME}")

if __name__ == "__main__":
    ingest_data()
