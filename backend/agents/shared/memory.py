import time
from typing import List, Dict, Any, Set, Optional
from backend.config import CONFIG

class Memory:
    """
    Memory system for tracking document snippets, reflections, and queries.
    
    This class provides methods for:
    - Tracking document snippets
    - Adding reflections on the research process
    - Tracking query evolution
    - Generating summaries of the research process
    """
    
    def __init__(self):
        """Initialize the memory system."""
        self.reset()
        
    def reset(self):
        """Reset all memory structures."""
        self.snips: List[Dict[str, Any]] = []
        self.visited: Set[str] = set()
        self.reflections: List[str] = []
        self.queries: List[Dict[str, str]] = []
        
    def track(self, text: str, doc_id: str, mode: str):
        """
        Track a document snippet.
        
        Args:
            text: The snippet text
            doc_id: The document ID
            mode: The slicing mode used
        """
        # Create a unique key for deduplication
        key = f"{doc_id}:{hash(text)}"
        
        # Skip if already visited
        if key in self.visited:
            return
            
        # Add to tracking structures
        self.visited.add(key)
        self.snips.append({"text": text, "source": doc_id, "mode": mode})
        
        # Enforce maximum snippets limit
        max_snippets = CONFIG.get("memory.max_snippets", 40)
        if len(self.snips) > max_snippets:
            self.snips.pop(0)
            
    def add_reflection(self, reflection: str):
        """
        Add a reflection on the research process.
        
        Args:
            reflection: The reflection text
        """
        self.reflections.append(reflection)
        
    def add_query(self, query: str, why: str = "initial"):
        """
        Track query evolution.
        
        Args:
            query: The query text
            why: Reason for the query change
        """
        self.queries.append({"q": query, "why": why})
        
    def summary(self) -> str:
        """
        Generate a summary of the research process.
        
        Returns:
            Markdown-formatted summary
        """
        s = "## Research Process\n"
        
        # Add query evolution if multiple queries
        if len(self.queries) > 1:
            s += "### Query evolution\n"
            s += "\n".join(f"- {q['q']} ({q['why']})" for q in self.queries) + "\n"
            
        # Add reflections if available
        if self.reflections:
            s += "### Reflections\n"
            s += "\n".join(f"- {r}" for r in self.reflections) + "\n"
            
        return s

# Create a singleton instance
MEM = Memory()
