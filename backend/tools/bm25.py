import json, pathlib
from whoosh.index import open_dir
from whoosh.qparser import QueryParser
from backend.config import CONFIG
from smolagents import tool

import logging
LOG = logging.getLogger(__name__)

# Load metadata produced by the indexer
META_PATH = CONFIG.get_path("indexing.meta")
META = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
#@tool
def bm25_search(query: str, k: int = 5):
    """
    Search documents using BM25 ranking algorithm.
    
    Args:
        query: The search query
        k: Maximum number of results to return
        
    Returns:
        List of document IDs matching the query
    
    Example
    -------
    >>> bm25_search("climate change", k=3)
    ['doc_123', 'doc_456', 'doc_789']
    """
    try:
        whoosh_dir = CONFIG.get_path("indexing.whoosh")
        if not whoosh_dir.exists():
            return [f"Error: Whoosh index not found at {whoosh_dir}"]
            
        ix = open_dir(str(whoosh_dir))
        qp = QueryParser("content", schema=ix.schema)
        q = qp.parse(query)
        
        with ix.searcher() as searcher:
            results = searcher.search(q, limit=k)
            return [hit["doc_id"] for hit in results]
    except Exception as e:
        return [f"Search error: {str(e)}"]
