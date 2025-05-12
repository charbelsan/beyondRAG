import json, pathlib, re
import fitz  # PyMuPDF
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
            # For PDF, use PyMuPDF to extract the text
            page_num = metadata.get("page_number", 0)
            try:
                with fitz.open(source) as pdf:
                    page_text = pdf[page_num].get_text("text")
                content = page_text
            except Exception as e:
                return f"Error extracting PDF content: {str(e)}"
        else:
            # For text files, we can read directly
            try:
                content = source_path.read_text(errors="ignore")
            except Exception as e:
                return f"Error reading file: {str(e)}"
        
        # Apply slicing based on mode
        result_content = ""
        if slice_mode == "auto":
            # Auto mode would use heuristics to determine the best slice
            # For simplicity, we'll just return the first N characters
            result_content = content[:slice_chars]
        elif slice_mode == "page":
            # For page mode, we return the whole page (already handled for PDF)
            result_content = content[:slice_chars]
        elif slice_mode == "paragraph":
            # For paragraph mode, find paragraph boundaries
            if source_path.suffix.lower() == ".pdf":
                paragraphs = re.split(r"\n{2,}", content)  # rough para split
                para_idx = metadata.get("para_idx", 0)
                result_content = paragraphs[para_idx][:slice_chars] if para_idx < len(paragraphs) else content[:slice_chars]
            else:
                paragraphs = content.split("\n\n")
                result_content = paragraphs[0][:slice_chars] if paragraphs else content[:slice_chars]
        elif slice_mode == "section":
            # For section mode, we would find section boundaries
            # This would require more sophisticated parsing
            result_content = content[:slice_chars]
        else:
            result_content = content[:slice_chars]
        
        # Include slice metadata in the snippet
        source_name = source_path.name
        page_info = f"p{metadata.get('page_number', 0)}" if source_path.suffix.lower() == ".pdf" else ""
        char_count = len(result_content)
        snippet_header = f"[{source_name} {page_info} • {char_count}c]\n"
        
        return snippet_header + result_content
    except Exception as e:
        return f"Error: {str(e)}"
