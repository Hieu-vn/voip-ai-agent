import httpx
import os
from qdrant_client import QdrantClient
from typing import List, Dict, Any

# Configuration
TEI_URL = os.getenv("TEI_URL", "http://localhost:8080")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "kb_vi"

# Initialize Qdrant client (assuming it's running)
cq = QdrantClient(host=QDRANT_URL.split("//")[-1].split(":")[0], port=int(QDRANT_URL.split(":")[-1]))

def embed(texts: list[str]) -> list[list[float]]:
    """Gets embeddings from the TEI service."""
    r = httpx.post(f"{TEI_URL}/embed", json={"inputs": texts})
    r.raise_for_status()
    return r.json()["embeddings"]

def tei_rerank(query: str, documents: list[str], top_k: int) -> list[str]:
    """
    Placeholder for reranking using TEI's reranker endpoint.
    For now, it's a mock that just returns the top_k documents.
    """
    # In a real implementation, this would call TEI's reranker endpoint
    # r = httpx.post(f"{TEI_URL}/rerank", json={"query": query, "texts": documents})
    # r.raise_for_status()
    # return [documents[x["index"]] for x in r.json()["results"]]
    return documents[:top_k]


def retrieve(query: str, k: int = 6) -> List[Dict[str, Any]]:
    """
    Retrieves relevant documents from Qdrant and reranks them.
    """
    query_vector = embed([query])[0]
    
    hits = cq.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=24, # Retrieve more for reranking
        score_threshold=0.2, # Adjust as needed
        # ef=128 # HNSW parameter, adjust based on performance
    )

    documents = [hit.payload["text"] for hit in hits]
    
    # Rerank the retrieved documents
    reranked_documents = tei_rerank(query, documents, top_k=k)

    return [{'text': doc} for doc in reranked_documents]
