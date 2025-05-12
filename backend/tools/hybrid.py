# File: backend/tools/hybrid.py

# --- Keep only non-circular top-level imports ---
from backend.config import CONFIG
import logging # Import logging for error handling
from smolagents import tool # Uncomment if decorating hybrid_search

# --- Remove problematic top-level imports ---
# from .bm25 import bm25_search  # REMOVED FROM TOP LEVEL
# from .vector import vector_search # REMOVED FROM TOP LEVEL


#@tool # Decorate if needed
def hybrid_search(query: str, k: int = 5, alpha: float = 0.5) -> list[str]:
    """
    Hybrid search combining BM25 and vector similarity with a weighted approach.

    Args:
        query: The search query
        k: Maximum number of results to return
        alpha: Weight for BM25 results (1-alpha for vector results). Range [0, 1].

    Returns:
        List of document IDs matching the query, or a list containing error messages.

    Example:
        hybrid_search("climate change impacts", k=3, alpha=0.7)
    """
    # ===> Import necessary functions *inside* the function <===
    from .bm25 import bm25_search
    from .vector import vector_search
    # ===========================================================

    try:
        # Validate inputs
        if not query:
            return ["Error: Empty query"]
        if k <= 0:
            return ["Error: Invalid value for k (must be > 0)"]
        if not 0 <= alpha <= 1: # Check range inclusive
            return ["Error: Invalid value for alpha (must be between 0 and 1)"]

        # Get results, requesting more for better ranking potential
        bm25_res = bm25_search(query, k=k * 2)
        vector_res = vector_search(query, k=k * 2)

        # --- Handle errors or empty results from individual searches ---
        bm25_error = bm25_res and isinstance(bm25_res[0], str) and bm25_res[0].startswith("Error:")
        vector_error = vector_res and isinstance(vector_res[0], str) and vector_res[0].startswith("Error:")

        if bm25_error and vector_error:
            logging.error(f"Both BM25 and Vector search failed. BM25: {bm25_res[0]}, Vector: {vector_res[0]}")
            return ["Error: Both BM25 and Vector search failed."]
        if bm25_error:
            logging.warning(f"BM25 search failed ({bm25_res[0]}), falling back to Vector search.")
            vector_res_final = vector_search(query, k=k) # Rerun with original k
            return vector_res_final if (vector_res_final and not (isinstance(vector_res_final[0], str) and vector_res_final[0].startswith("Error:"))) else ["No matching documents found (Vector only)."]
        if vector_error:
            logging.warning(f"Vector search failed ({vector_res[0]}), falling back to BM25 search.")
            bm25_res_final = bm25_search(query, k=k) # Rerun with original k
            return bm25_res_final if (bm25_res_final and not (isinstance(bm25_res_final[0], str) and bm25_res_final[0].startswith("Error:"))) else ["No matching documents found (BM25 only)."]

        # Filter out potential non-string results if necessary
        bm25_docs = [d for d in bm25_res if isinstance(d, str)]
        vector_docs = [d for d in vector_res if isinstance(d, str)]

        if not bm25_docs and not vector_docs:
             return ["No matching documents found."]
        # --- End Handle errors ---

        # Combine results (e.g., using Reciprocal Rank Fusion inspired scoring)
        scores: dict[str, float] = {}
        fusion_k = 60 # Constant for RRF scoring

        for rank, doc_id in enumerate(bm25_docs):
            scores[doc_id] = scores.get(doc_id, 0) + alpha * (1 / (rank + fusion_k))

        for rank, doc_id in enumerate(vector_docs):
            scores[doc_id] = scores.get(doc_id, 0) + (1 - alpha) * (1 / (rank + fusion_k))

        # Sort by combined score
        sorted_results = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        # Return top k document IDs
        final_doc_ids = [doc_id for doc_id, score in sorted_results[:k]]

        return final_doc_ids if final_doc_ids else ["No matching documents found after hybrid scoring."]

    except Exception as e:
        logging.error(f"Hybrid search failed unexpectedly: {e}", exc_info=True) # Log full traceback
        return [f"Hybrid search error: {str(e)}"] # Return error as list item

