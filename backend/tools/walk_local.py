import networkx as nx, pathlib, json


# robust gpickle helpers (works on any NetworkX 2.x/3.x)
try:                                   # ≥ 3.2
    from networkx.readwrite.gpickle import read_gpickle, write_gpickle  # type: ignore
except ImportError:                    # ≤ 3.1
    import pickle, pathlib
    def read_gpickle(path):            # type: ignore
        with open(path, "rb") as f:
            return pickle.load(f)
    def write_gpickle(G, path):        # type: ignore
        with open(path, "wb") as f:
            pickle.dump(G, f)

from backend.config import CONFIG

# Load graph and map from configuration
GRAPH_PATH = CONFIG.get_path("indexing.graph")
MAP_PATH = CONFIG.get_path("indexing.map")

# Initialize graph and map
G = read_gpickle(GRAPH_PATH) if GRAPH_PATH.exists() else nx.DiGraph()
MAP = json.loads(MAP_PATH.read_text()) if MAP_PATH.exists() else []

def walk_local(doc_id: str, direction: str = "next", k: int = 1):
    """
    Walk the document graph to find related documents.
    
    Args:
        doc_id: The starting document ID
        direction: Direction to walk ('next', 'prev', 'in_folder', 'contains')
        k: Maximum number of results to return
        
    Returns:
        List of related document IDs
        
    Example
    -------
    >>> walk_local("doc_123", direction="next", k=2)
    ['doc_124', 'doc_125']
    """
    try:
        if not GRAPH_PATH.exists():
            return [f"Error: Graph file not found at {GRAPH_PATH}"]
            
        if doc_id not in G:
            return [f"Error: Document ID '{doc_id}' not found in graph"]
            
        # Get neighbors based on direction
        if direction == "next":
            # Forward navigation
            neighbors = [n for n in G.neighbors(doc_id) if G.get_edge_data(doc_id, n).get('type') == 'next']
        elif direction == "prev":
            # Backward navigation
            neighbors = [n for n in G.neighbors(doc_id) if G.get_edge_data(doc_id, n).get('type') == 'prev']
        elif direction == "in_folder":
            # Document to folder navigation
            neighbors = [n for n in G.neighbors(doc_id) if G.get_edge_data(doc_id, n).get('type') == 'in_folder']
        elif direction == "contains":
            # Folder to document navigation
            neighbors = [n for n in G.neighbors(doc_id) if G.get_edge_data(doc_id, n).get('type') == 'contains']
        else:
            # Default: all neighbors
            neighbors = list(G.neighbors(doc_id))
            
        return neighbors[:k]
    except Exception as e:
        return [f"Error walking graph: {str(e)}"]
