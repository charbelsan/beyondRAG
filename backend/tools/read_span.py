import json, pathlib
from backend.config import CONFIG

# Load metadata
META_PATH = CONFIG.get_path("indexing.meta")
META = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}

# Get default slice settings from config
DEFAULT_CHARS = CONFIG.get("slice.default_chars", 750)
DEFAULT_MODE = CONFIG.get("slice.default_mode", "auto")

def read_span(doc_id: str, mode: str = None, chars: int = None):
    """
    Read a span of text from a document.
    
    Args:
        doc_id: The document ID to read from
        mode: The slicing mode ('auto', 'page', 'paragraph', 'section')
        chars: The number of characters to return
        
    Returns:
        The text content of the specified span
    
    Example
    -------
    >>> read_span("doc_123", mode="paragraph", chars=500)
    'This is the content of the paragraph...'
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
            
        source_path = pathlib.Path(source)
        if not source_path.exists():
            return f"Error: Source file '{source}' not found"
            
        # Determine slicing mode
        slice_mode = mode if mode else DEFAULT_MODE
        slice_chars = chars if chars else DEFAULT_CHARS
        
        # Get content based on document type
        content = ""
        source_path = pathlib.Path(source)
        
        if source_path.suffix.lower() == ".pdf":
            # For PDF, we would use the page number from metadata
            page_num = metadata.get("page_number")
            if page_num is not None:
                # In a real implementation, we would use PyMuPDF to extract the text
                # For now, we'll just return a placeholder
                content = f"[PDF content from {source}, page {page_num}]"
            else:
                content = f"[PDF content from {source}]"
        else:
            # For text files, we can read directly
            try:
                content = source_path.read_text(errors="ignore")
            except Exception as e:
                return f"Error reading file: {str(e)}"
        
        # Apply slicing based on mode
        if slice_mode == "auto":
            # Auto mode would use heuristics to determine the best slice
            # For simplicity, we'll just return the first N characters
            return content[:slice_chars]
        elif slice_mode == "page":
            # For page mode, we return the whole page (already handled for PDF)
            return content[:slice_chars]
        elif slice_mode == "paragraph":
            # For paragraph mode, we would find paragraph boundaries
            # For simplicity, we'll split by double newlines and return the first paragraph
            paragraphs = content.split("\n\n")
            return paragraphs[0][:slice_chars] if paragraphs else content[:slice_chars]
        elif slice_mode == "section":
            # For section mode, we would find section boundaries
            # This would require more sophisticated parsing
            return content[:slice_chars]
        else:
            return content[:slice_chars]
    except Exception as e:
        return f"Error: {str(e)}"
