"""
backend/tools/research_tools.py
-------------------------------
Advanced research tools for the code agent to implement the full research loop:
PLAN → SEARCH → READ → REFLECT → NAVIGATE → SYNTHESIZE

These tools allow the code agent to perform sophisticated research operations.
"""

from __future__ import annotations

import json
import pathlib
from typing import Dict, List, Any, Optional

from smolagents import tool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from backend.config import CONFIG
from backend.agents.shared.memory import MEM
from backend.tools.bm25 import bm25_search
from backend.tools.vector import vector_search
from backend.tools.read_span import read_span

# Import graph utilities
try:
    from networkx.readwrite.gpickle import read_gpickle
except ImportError:
    from indexing.graph import _pickle_read as read_gpickle

# Initialize LLM for tool operations
LLM = ChatOpenAI(
    base_url=CONFIG.get("llm.base_url", "http://localhost:11434/v1"),
    api_key=CONFIG.get("llm.api_key", "ollama"),
    model_name=CONFIG.get("llm.model_id", "qwen2.5:32b-instruct"),
    temperature=0.2,
)

# Helper to extract document IDs from search results
def _extract_ids(hits) -> list[str]:
    ids = []
    for h in hits or []:
        if isinstance(h, tuple):
            ids.append(str(h[0]))
        elif isinstance(h, dict) and "id" in h:
            ids.append(str(h["id"]))
        else:
            ids.append(str(h))
    return ids

# ── Research Tools ──────────────────────────────────────────────────────────

@tool
def plan_research(question: str) -> str:
    """
    Create a research plan for answering the question.
    
    Args:
        question: The user's question
        
    Returns:
        A structured research plan
    
    Example:
    >>> plan_research("What is the CLIP architecture?")
    "1. Identify key components of CLIP\n2. Understand how image and text encoders work\n3. ..."
    """
    prompt = PromptTemplate(
        template="""
        You are tasked with creating a research plan to answer the following question:
        {question}
        
        Create a step-by-step plan that outlines:
        1. The key concepts that need to be understood
        2. The specific information that needs to be gathered
        3. The potential relationships between concepts that should be explored
        4. The order in which to explore these concepts
        
        Your plan should be structured, concise, and focused on efficiently answering the question.
        """,
        input_variables=["question"],
    )
    
    llm_input = prompt.format(question=question)
    plan = LLM.invoke(llm_input).content.strip()
    return plan

@tool
def hybrid_search_with_content(query: str, k: int = 5, visited_docs: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Perform a hybrid search (BM25 + vector) and return both document IDs and content.
    
    Args:
        query: The search query
        k: Maximum number of results to return
        visited_docs: Optional list of already visited document IDs to exclude
        
    Returns:
        Dictionary with document IDs, content snippets, and tracking info
    
    Example:
    >>> hybrid_search_with_content("CLIP architecture", k=3)
    {"doc_ids": ["doc_123", "doc_456"], "snippets": ["CLIP uses a...", "The architecture consists of..."], "new_docs_found": 2}
    """
    # Perform searches
    bm25_ids = _extract_ids(bm25_search(query, k=k))
    vector_ids = _extract_ids(vector_search(query, k=k))
    top_ids = list(dict.fromkeys(bm25_ids + vector_ids))[:k]
    
    # Filter out already visited documents if provided
    if visited_docs:
        new_ids = [did for did in top_ids if did not in visited_docs]
    else:
        new_ids = top_ids
    
    # Get content for each document
    snippets = []
    retrieved_ids = []
    meta_path = CONFIG.get_path("indexing.meta")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    
    for did in new_ids:
        txt = read_span(did, mode="auto", chars=750)
        if not txt:  # fallback to stored raw_text
            txt = meta.get(did, {}).get("raw_text", "")
        if txt:
            snippets.append(txt)
            retrieved_ids.append(did)
            MEM.track(txt, did, mode="auto")
    
    return {
        "doc_ids": retrieved_ids,
        "snippets": snippets,
        "new_docs_found": len(snippets)
    }

@tool
def analyze_content(question: str, content: List[str]) -> str:
    """
    Analyze content snippets in relation to the research question.
    
    Args:
        question: The research question
        content: List of content snippets to analyze
        
    Returns:
        Structured analysis of the content
    
    Example:
    >>> analyze_content("What is CLIP?", ["CLIP (Contrastive Language-Image Pre-training) is a neural network..."])
    "The content explains that CLIP is a neural network that..."
    """
    if not content:
        return "No content provided for analysis."
    
    prompt = PromptTemplate(
        template="""
        Analyze the following content in relation to this question:
        {question}
        
        Content:
        {content}
        
        Extract and organize the key information from this content:
        1. What are the main concepts discussed?
        2. What specific facts or details are provided?
        3. How does this information relate to the question?
        4. What is the significance of this information?
        
        Provide a structured analysis that captures the essence of this content.
        """,
        input_variables=["question", "content"],
    )
    
    content_text = "\n\n".join(content)
    llm_input = prompt.format(question=question, content=content_text)
    analysis = LLM.invoke(llm_input).content.strip()
    return analysis

@tool
def reflect_on_research(question: str, findings: str, plan: Optional[str] = None) -> Dict[str, Any]:
    """
    Reflect on the current research state and identify gaps.
    
    Args:
        question: The original question
        findings: Current research findings and analysis
        plan: Optional research plan
        
    Returns:
        Dictionary with reflection and potentially reformulated query
    
    Example:
    >>> reflect_on_research("How does CLIP work?", "CLIP consists of two encoders...", "1. Understand encoders...")
    {"reflection": "We've learned about the encoders but still need to understand training...", 
     "reformulated_query": "CLIP training process", "sufficient_info": False}
    """
    prompt = PromptTemplate(
        template="""
        Question: {question}
        
        Research plan: {plan}
        
        Current findings:
        {findings}
        
        Reflect deeply on the current state of research:
        1. What have we learned that directly addresses the question?
        2. What critical information is still missing?
        3. Are there any contradictions or uncertainties in the information gathered?
        4. What specific aspect should we focus on next?
        5. How should we adjust our search to find the missing information?
        
        If we need to reformulate our search query, provide a "Reformulated query: [new query]" on a separate line.
        
        Finally, indicate whether we have "Sufficient information: [Yes/No]" to answer the question.
        """,
        input_variables=["question", "findings", "plan"],
    )
    
    llm_input = prompt.format(
        question=question,
        findings=findings,
        plan=plan or "No explicit plan."
    )
    
    reflection = LLM.invoke(llm_input).content.strip()
    MEM.add_reflection(reflection)
    
    # Extract reformulated query if present
    reformulated_query = None
    if "reformulated query:" in reflection.lower():
        query_parts = reflection.lower().split("reformulated query:")
        if len(query_parts) > 1:
            reformulated_query = query_parts[1].strip().split("\n")[0].strip()
            MEM.add_query(reformulated_query, "implicit_reformulation")
    
    # Determine if we have sufficient information
    sufficient_info = False
    if "sufficient information:" in reflection.lower():
        info_parts = reflection.lower().split("sufficient information:")
        if len(info_parts) > 1:
            sufficient_text = info_parts[1].strip().split("\n")[0].strip()
            sufficient_info = sufficient_text.startswith("yes")
    
    return {
        "reflection": reflection,
        "reformulated_query": reformulated_query or question,
        "sufficient_info": sufficient_info
    }

@tool
def navigate_document_graph(doc_id: str, visited_docs: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Navigate to related documents using the document graph.
    
    Args:
        doc_id: The document ID to find relations for
        visited_docs: Optional list of already visited document IDs to exclude
        
    Returns:
        Dictionary with related document content and metadata
    
    Example:
    >>> navigate_document_graph("doc_123", ["doc_123"])
    {"related_docs": ["doc_456"], "snippets": ["Related content..."], "relations": ["next_page"]}
    """
    if not doc_id:
        return {"error": "No document ID provided."}
    
    # Initialize visited docs if not provided
    if visited_docs is None:
        visited_docs = []
    
    # Load the graph
    graph_path = CONFIG.get_path("indexing.graph")
    if not graph_path.exists():
        return {"error": "Document graph not found."}
    
    try:
        G = read_gpickle(graph_path)
        
        if doc_id not in G:
            return {"error": f"Document {doc_id} not found in graph."}
        
        # Find related documents
        related_docs = []
        relation_types = []
        
        for neighbor in G.neighbors(doc_id):
            if neighbor not in visited_docs and not neighbor.startswith('/'):
                edge_data = G.get_edge_data(doc_id, neighbor)
                relation_type = edge_data.get('type', 'related')
                related_docs.append(neighbor)
                relation_types.append(relation_type)
        
        if not related_docs:
            return {"error": "No unvisited related documents found."}
        
        # Get content from related documents
        snippets = []
        meta_path = CONFIG.get_path("indexing.meta")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        
        for i, did in enumerate(related_docs[:3]):  # Limit to 3 related docs
            txt = read_span(did, mode="auto", chars=750)
            if not txt:  # fallback to stored raw_text
                txt = meta.get(did, {}).get("raw_text", "")
            if txt:
                relation = relation_types[i] if i < len(relation_types) else "related"
                snippet_with_relation = f"[Relation: {relation}]\n{txt}"
                snippets.append(snippet_with_relation)
                MEM.track(txt, did, mode="auto")
        
        return {
            "related_docs": related_docs[:len(snippets)],
            "snippets": snippets,
            "relations": relation_types[:len(snippets)]
        }
    
    except Exception as e:
        return {"error": f"Error navigating document graph: {str(e)}"}

@tool
def synthesize_answer(question: str, findings: str, reflections: Optional[str] = None) -> str:
    """
    Synthesize a comprehensive answer from research findings.
    
    Args:
        question: The original question
        findings: The research findings and analysis
        reflections: Optional reflections on the research process
        
    Returns:
        A comprehensive answer to the question
    
    Example:
    >>> synthesize_answer("How does CLIP work?", "CLIP consists of two encoders...", "We found detailed information...")
    "CLIP (Contrastive Language-Image Pre-training) works by using two encoders..."
    """
    prompt = PromptTemplate(
        template="""
        Question: {question}
        
        Research findings:
        {findings}
        
        Research reflections:
        {reflections}
        
        Synthesize a comprehensive answer that:
        1. Directly addresses the original question
        2. Integrates all relevant information discovered
        3. Acknowledges any limitations or uncertainties
        4. Is well-structured and logically organized
        5. Cites specific sources where appropriate
        
        Your answer should be thorough yet concise, focusing on providing maximum value to the user.
        """,
        input_variables=["question", "findings", "reflections"],
    )
    
    llm_input = prompt.format(
        question=question,
        findings=findings,
        reflections=reflections or "No explicit reflections."
    )
    
    answer = LLM.invoke(llm_input).content.strip()
    return answer