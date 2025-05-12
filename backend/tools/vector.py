import json, pathlib, numpy as np, faiss
from sentence_transformers import SentenceTransformer
from backend.config import CONFIG

# Load metadata and model
META_PATH = CONFIG.get_path("indexing.meta")
MAP_PATH = CONFIG.get_path("indexing.map")
FAISS_PATH = CONFIG.get_path("indexing.faiss")

META = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
MAP = json.loads(MAP_PATH.read_text()) if MAP_PATH.exists() else []

# Initialize the model - use the same model as in the indexer
MODEL = SentenceTransformer(CONFIG.get("chunking.semantic.embeddings_model", "all-mpnet-base-v2"))

def vector_search(query: str, k: int = 5):
    """
    Search documents using vector similarity.
    
    Args:
        query: The search query
        k: Maximum number of results to return
        
    Returns:
        List of document IDs matching the query
    
    Example
    -------
    >>> vector_search("climate change impacts", k=3)
    ['doc_123', 'doc_456', 'doc_789']
    """
    try:
        if not FAISS_PATH.exists():
            return [f"Error: Vector index not found at {FAISS_PATH}"]
            
        if not MAP_PATH.exists():
            return [f"Error: Document map not found at {MAP_PATH}"]
            
        # Encode the query
        query_vector = MODEL.encode(query).astype("float32").reshape(1, -1)
        
        # Load the FAISS index
        index = faiss.read_index(str(FAISS_PATH))
        
        # Search
        distances, indices = index.search(query_vector, k)
        
        # Map indices to document IDs
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(MAP):
                results.append(MAP[idx])
            
        return results
    except Exception as e:
        return [f"Search error: {str(e)}"]
