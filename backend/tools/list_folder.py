import pathlib, json
from backend.config import CONFIG
from smolagents import tool

# Load metadata produced by the indexer
META_PATH = CONFIG.get_path("indexing.meta")
META = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
#@tool
def list_folder(folder_path: str, k: int = 20):
    """
    Return up to *k* doc-IDs whose source files live **directly**
    inside *folder_path* (no recursive walk).

    Args:
        folder_path: The folder path to list documents from
        k: Maximum number of document IDs to return
        
    Returns:
        List of document IDs from the specified folder
        
    Example
    -------
    >>> list_folder("/app/docs/contracts", k=5)
    ['contracts_01_page_0', 'contracts_02_page_0', ...]
    """
    try:
        if not META_PATH.exists():
            return [f"Error: Metadata file not found at {META_PATH}"]
            
        folder = pathlib.Path(folder_path).as_posix().rstrip("/")

        out = []
        for did, m in META.items():
            if "source" in m and pathlib.Path(m["source"]).parent.as_posix() == folder:
                out.append(did)
                if len(out) >= k:
                    break
                    
        if not out:
            return [f"No documents found in folder: {folder_path}"]
            
        return out
    except Exception as e:
        return [f"Error listing folder: {str(e)}"]
