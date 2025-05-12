import json, pathlib
from .read_span import read_span
from backend.config import CONFIG

# Load metadata
META_PATH = CONFIG.get_path("indexing.meta")
META = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}

def text_browser(doc_id: str, direction: str = "next", k: int = 1):
    """
    Browse text documents by navigating between related spans.
    
    Args:
        doc_id: The current document ID
        direction: Navigation direction ('next', 'prev', 'parent', 'child')
        k: Number of steps to take in the specified direction
        
    Returns:
        The content of the target document after navigation
    
    Example
    -------
    >>> text_browser("doc_123", direction="next")
    'Content of the next document...'
    """
    try:
        if not META_PATH.exists():
            return f"Error: Metadata file not found at {META_PATH}"
            
        if doc_id not in META:
            return f"Error: Document ID '{doc_id}' not found"
            
        # Get document metadata
        metadata = META[doc_id]
        source = metadata.get("source")
        
        if not source:
            return f"Error: Source not specified for '{doc_id}'"
            
        # Handle navigation based on direction
        target_id = doc_id
        
        if direction == "next" or direction == "prev":
            # For next/prev, we use page numbers for PDFs
            page_num = metadata.get("page_number")
            if page_num is not None:
                # Calculate target page number
                target_page = page_num + k if direction == "next" else page_num - k
                
                # Construct target document ID
                target_id = f"{source}::page_{target_page}"
                
                # Check if target exists
                if target_id not in META:
                    return f"Error: No {direction} document found for '{doc_id}'"
            else:
                return f"Error: Cannot navigate {direction} for non-paginated document '{doc_id}'"
        elif direction == "parent":
            # For parent navigation, we would need a hierarchical structure
            # This is a simplified implementation
            return f"Error: Parent navigation not implemented for '{doc_id}'"
        elif direction == "child":
            # For child navigation, we would need a hierarchical structure
            # This is a simplified implementation
            return f"Error: Child navigation not implemented for '{doc_id}'"
        else:
            return f"Error: Unknown direction '{direction}'"
            
        # Read the content of the target document
        return read_span(target_id)
    except Exception as e:
        return f"Error: {str(e)}"
