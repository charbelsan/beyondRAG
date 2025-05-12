from typing import List, Dict, Any, Optional
from backend.config import CONFIG

try:
    from gpt_researcher.publisher import publish_report as gpt_publish
    
    def publish(snippets: List[Dict[str, Any]]) -> str:
        """
        Format document snippets as a markdown report.
        
        Args:
            snippets: List of document snippets with metadata
            
        Returns:
            Markdown-formatted report
        """
        try:
            # Use gpt-researcher's publisher if available
            return gpt_publish(snippets, output_format="markdown")
        except Exception as e:
            print(f"Error using gpt-researcher publisher: {str(e)}")
            # Fall back to simple formatter
            return _simple_formatter(snippets)
except ImportError:
    # If gpt-researcher is not available, use a simple formatter
    def publish(snippets: List[Dict[str, Any]]) -> str:
        """
        Format document snippets as a markdown report.
        
        Args:
            snippets: List of document snippets with metadata
            
        Returns:
            Markdown-formatted report
        """
        return _simple_formatter(snippets)

def _simple_formatter(snippets: List[Dict[str, Any]]) -> str:
    """
    Simple formatter for document snippets.
    
    Args:
        snippets: List of document snippets with metadata
        
    Returns:
        Markdown-formatted report
    """
    if not snippets:
        return "No sources found."
        
    result = []
    for i, snip in enumerate(snippets):
        source = snip.get("source", "Unknown source")
        mode = snip.get("mode", "auto")
        text = snip.get("text", "").strip()
        
        if not text:
            continue
            
        # Format the snippet
        result.append(f"**Source {i+1}**: {source} (mode: {mode})")
        result.append(f"```\n{text[:500]}{'...' if len(text) > 500 else ''}\n```")
        result.append("")
        
    return "\n".join(result)
