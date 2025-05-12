"""
backend/agents/reasoning_agent.py
---------------------------------
Enhanced LangGraph‑based reasoning agent with full research loop:
PLAN → SEARCH → READ → REFLECT → NAVIGATE → SYNTHESIZE

This implementation handles reformulation implicitly during the reflection phase.
"""

from __future__ import annotations

import os
import json
import pathlib
import logging
import re
from typing import List, Optional, Dict, Any, Callable

from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("reasoning_agent")

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from backend.tools.bm25 import bm25_search
from backend.tools.vector import vector_search
from backend.tools.read_span import read_span
from backend.agents.shared.memory import MEM
from backend.config import CONFIG

# Import graph utilities
try:
    from networkx.readwrite.gpickle import read_gpickle
except ImportError:
    from indexing.graph import _pickle_read as read_gpickle

# ── LLM ---------------------------------------------------------------------
LLM = ChatOpenAI(
    base_url=os.getenv("OPENAI_API_BASE", CONFIG.get("llm.base_url", "http://localhost:11434/v1")),
    api_key=os.getenv("OPENAI_API_KEY", CONFIG.get("llm.api_key", "ollama")),
    model_name=os.getenv("OPENAI_MODEL_NAME", CONFIG.get("llm.model_id", "qwen2.5:32b-instruct")),
    temperature=0.2,
)

# ── regex patterns for parsing control lines ---------------------------------
TAG_NEED_MORE = re.compile(r"need[- ]more[- ]search:\s*(yes|no)", re.I)
TAG_COVERAGE  = re.compile(r"coverage:\s*(high|partial|low)", re.I)
TAG_NEWQUERY  = re.compile(r"new[- ]query:\s*(.+)", re.I)

# ── enhanced state schema ---------------------------------------------------
class ResearchState(BaseModel):
    question: str
    plan: Optional[str] = None
    context: List[str] = Field(default_factory=list)
    reflections: List[str] = Field(default_factory=list)
    visited_docs: List[str] = Field(default_factory=list)
    current_focus: Optional[str] = None
    current_query: str = ""  # Will be updated during reflection
    iterations: int = 0  # Counter to prevent infinite loops
    coverage: str = "partial"  # high/partial/low from LLM assessment
    need_more: bool = True  # Flag from LLM to continue or stop
    previous_queries: List[str] = Field(default_factory=list)  # Track queries to avoid duplicates
    answer: Optional[str] = None

# ── helper to normalise hit shapes ------------------------------------------
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

# ── prompt templates --------------------------------------------------------
_PLAN_TEMPLATE = PromptTemplate(
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

_READ_TEMPLATE = PromptTemplate(
    template="""
Analyze the following content in relation to this question:
{question}

Content:
{context}

Extract and organize the key information from this content:
1. What are the main concepts discussed?
2. What specific facts or details are provided?
3. How does this information relate to the question?
4. What is the significance of this information?

Provide a structured analysis that captures the essence of this content.

Finally, on one line only, state:
Coverage: high (if the content fully answers the question)
Coverage: partial (if the content partially answers the question)
Coverage: low (if the content is not relevant to the question)
""",
    input_variables=["question", "context"],
)

_REFLECT_TEMPLATE = PromptTemplate(
    template="""
Question: {question}

Research plan: {plan}

Information gathered so far:
{context}

Current understanding:
{current_focus}

Previous reflections:
{reflections}

Reflect deeply on the current state of research:
1. What have we learned that directly addresses the question?
2. What critical information is still missing?
3. Are there any contradictions or uncertainties in the information gathered?
4. What specific aspect should we focus on next?
5. How should we adjust our search to find the missing information?

If you need more information, output exactly:
Need-more-search: yes

Otherwise output exactly:
Need-more-search: no

If another search is needed, also provide one line:
New-query: <max-5-keywords, lowercase, no punctuation>

Examples
Need-more-search: no
Need-more-search: yes
New-query: clip architecture vit

Do not output anything else on those lines.
""",
    input_variables=["question", "plan", "context", "current_focus", "reflections"],
)

_SYNTH_TEMPLATE = PromptTemplate(
    template="""
Question: {question}

Research plan: {plan}

All gathered information:
{context}

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
    input_variables=["question", "plan", "context", "reflections"],
)

# ── node: plan --------------------------------------------------------------
def plan_node(state: ResearchState) -> dict:
    """Develop a research plan for answering the question."""
    logger.info(f"PLAN: Creating research plan for question: {state.question[:50]}...")
    
    llm_input = _PLAN_TEMPLATE.format(question=state.question)
    plan = LLM.invoke(llm_input).content.strip()
    
    logger.info(f"PLAN: Created plan with {len(plan.split())} words")
    
    # Make the first query automatically terse (max 4 tokens, no question mark)
    first_query = " ".join(state.question.lower().replace("?", "").split()[:4])
    logger.info(f"PLAN: Initial terse query: {first_query}")
    
    # Add to previous queries list
    state.previous_queries.append(first_query)
    
    return {
        "plan": plan,
        "current_query": first_query
    }

# ── node: search (enhanced version of retrieve) -----------------------------
def search_node(state: ResearchState) -> dict:
    """Search for relevant documents using the current query."""
    query = state.current_query or state.question
    
    # Guard against pointless reformulations
    if query in state.previous_queries[:-1]:  # Skip the last one which we just added
        logger.info(f"SEARCH: Duplicate query detected – skipping search: {query[:50]}")
        return {}
    
    logger.info(f"SEARCH: Searching with query: {query[:50]}...")
    
    bm25_ids = _extract_ids(bm25_search(query, k=5))
    logger.info(f"SEARCH: BM25 returned {len(bm25_ids)} results")
    
    vector_ids = _extract_ids(vector_search(query, k=5))
    logger.info(f"SEARCH: Vector search returned {len(vector_ids)} results")
    
    top_ids = list(dict.fromkeys(bm25_ids + vector_ids))[:5]
    
    # Filter out already visited documents
    new_ids = [did for did in top_ids if did not in state.visited_docs]
    logger.info(f"SEARCH: Found {len(new_ids)} new documents to explore")
    
    snippets = []
    meta_path = CONFIG.get_path("indexing.meta")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    
    for did in new_ids:
        txt = read_span(did, mode="auto", chars=750)
        if not txt:  # fallback to stored raw_text
            txt = meta.get(did, {}).get("raw_text", "")
        if txt:
            snippets.append(txt)
            MEM.track(txt, did, mode="auto")
            state.visited_docs.append(did)
    
    logger.info(f"SEARCH: Retrieved {len(snippets)} text snippets")
    return {"context": state.context + snippets}

# ── node: read --------------------------------------------------------------
def read_node(state: ResearchState) -> dict:
    """Analyze retrieved content and extract key information."""
    logger.info(f"READ: Analyzing {len(state.context)} context items")
    
    if not state.context:
        logger.info("READ: No content available to analyze")
        return {"current_focus": "No content available to analyze.", "coverage": "low"}
    
    # Use only the most recent context items for analysis
    recent_context = state.context[-3:] if len(state.context) > 3 else state.context
    context_text = "\n\n".join(recent_context)
    
    llm_input = _READ_TEMPLATE.format(
        question=state.question,
        context=context_text
    )
    analysis = LLM.invoke(llm_input).content.strip()
    
    # Extract coverage information
    cov = TAG_COVERAGE.search(analysis)
    coverage = cov.group(1).lower() if cov else "partial"
    logger.info(f"READ: Content coverage assessment: {coverage}")
    
    logger.info(f"READ: Analysis complete with {len(analysis.split())} words")
    return {"current_focus": analysis, "coverage": coverage}

# ── node: reflect -----------------------------------------------------------
def reflect_node(state: ResearchState) -> dict:
    """Reflect on current knowledge and identify gaps, implicitly reformulating if needed."""
    logger.info(f"REFLECT: Iteration {state.iterations + 1}, reflecting on current knowledge")
    
    # Prepare reflections text
    reflections_text = "\n".join(state.reflections) if state.reflections else "No previous reflections."
    
    # Prepare context summary (limit to avoid token overflow)
    context_summary = "\n\n".join(state.context[-5:]) if state.context else "No context gathered yet."
    
    llm_input = _REFLECT_TEMPLATE.format(
        question=state.question,
        plan=state.plan or "No explicit plan.",
        context=context_summary,
        current_focus=state.current_focus or "No current focus.",
        reflections=reflections_text
    )
    
    reflection = LLM.invoke(llm_input).content.strip()
    MEM.add_reflection(reflection)
    
    # Extract need-more-search flag
    need = TAG_NEED_MORE.search(reflection)
    need_more = (need.group(1).lower() == "yes") if need else True
    logger.info(f"REFLECT: Need more search: {need_more}")
    
    # Log the actual match for debugging
    if need:
        logger.info(f"REFLECT: Need-more-search match: '{need.group(0)}' -> '{need.group(1)}'")
    
    # Extract new query if present
    new_query = state.current_query  # Default to current query
    nq = TAG_NEWQUERY.search(reflection)
    if nq:
        new_query = nq.group(1).strip()
        if new_query not in state.previous_queries:
            state.previous_queries.append(new_query)
            MEM.add_query(new_query, "explicit_reformulation")
            logger.info(f"REFLECT: Query reformulated to: {new_query[:50]}...")
    
    logger.info(f"REFLECT: Added reflection with {len(reflection.split())} words")
    return {
        "reflections": state.reflections + [reflection],
        "current_query": new_query,
        "need_more": need_more,
        "iterations": state.iterations + 1  # Increment iteration counter
    }

# ── node: navigate ----------------------------------------------------------
def navigate_node(state: ResearchState) -> dict:
    """Navigate to related documents using the graph."""
    if not state.visited_docs:
        return {}
    
    # Get the last visited document
    last_doc = state.visited_docs[-1]
    
    # Load the graph
    graph_path = CONFIG.get_path("indexing.graph")
    if not graph_path.exists():
        return {"current_focus": state.current_focus + "\n\nNo document graph available for navigation."}
    
    try:
        G = read_gpickle(graph_path)
        
        # Find related documents
        related_docs = []
        if last_doc in G:
            for neighbor in G.neighbors(last_doc):
                if neighbor not in state.visited_docs and not neighbor.startswith('/'):
                    edge_data = G.get_edge_data(last_doc, neighbor)
                    relation_type = edge_data.get('type', 'related')
                    related_docs.append((neighbor, relation_type))
        
        if not related_docs:
            return {"current_focus": state.current_focus + "\n\nNo related documents found."}
        
        # Get content from related documents
        new_snippets = []
        meta_path = CONFIG.get_path("indexing.meta")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        
        for did, relation in related_docs[:3]:  # Limit to 3 related docs
            txt = read_span(did, mode="auto", chars=750)
            if not txt:  # fallback to stored raw_text
                txt = meta.get(did, {}).get("raw_text", "")
            if txt:
                snippet_with_relation = f"[Relation: {relation}]\n{txt}"
                new_snippets.append(snippet_with_relation)
                MEM.track(txt, did, mode="auto")
                state.visited_docs.append(did)
        
        return {"context": state.context + new_snippets}
    
    except Exception as e:
        return {"current_focus": state.current_focus + f"\n\nError navigating document graph: {e}"}

# ── node: synthesize (enhanced version of synthesise) -----------------------
def synthesize_node(state: ResearchState) -> dict:
    """Synthesize final answer from all gathered information."""
    logger.info(f"SYNTHESIZE: Creating final answer after {state.iterations} iterations")
    logger.info(f"SYNTHESIZE: Using {len(state.context)} context items and {len(state.reflections)} reflections")
    
    ctx = "\n\n".join(state.context) or "[no useful context]"
    reflections = "\n".join(state.reflections) or "[no reflections]"
    
    llm_input = _SYNTH_TEMPLATE.format(
        question=state.question,
        plan=state.plan or "No explicit plan.",
        context=ctx,
        reflections=reflections
    )
    
    answer = LLM.invoke(llm_input).content.strip()
    logger.info(f"SYNTHESIZE: Generated answer with {len(answer.split())} words")
    return {"answer": answer}

# ── edge conditions --------------------------------------------------------
def should_navigate(state: ResearchState) -> bool:
    """Determine if graph navigation is needed."""
    if not state.reflections:
        return False
    
    last_reflection = state.reflections[-1].lower()
    navigation_indicators = [
        "related document",
        "more context",
        "adjacent page",
        "next page",
        "previous page",
        "navigate to",
        "explore related"
    ]
    
    return any(indicator in last_reflection for indicator in navigation_indicators)

def has_sufficient_info(state: ResearchState) -> bool:
    """Determine if we have sufficient information to answer."""
    # Use the LLM's own assessment
    sufficient = (state.coverage == "high") or (not state.need_more)
    
    if sufficient:
        logger.info(f"Sufficient information detected: coverage={state.coverage}, need_more={state.need_more}")
    
    return sufficient

def needs_more_search(state: ResearchState) -> bool:
    """Determine if more searching is needed."""
    # Default to True if no reflections yet
    if not state.reflections:
        return True
    
    # If we have reflections, check the latest one
    last_reflection = state.reflections[-1].lower()
    more_search_indicators = [
        "need more information",
        "insufficient context",
        "more details needed",
        "search for",
        "look for",
        "find more",
        "reformulated query:"
    ]
    
    return any(indicator in last_reflection for indicator in more_search_indicators)

# ── build enhanced graph ---------------------------------------------------
def build_graph():
    """Build and return the enhanced research graph."""
    graph = StateGraph(state_schema=ResearchState)
    
    # Add nodes - use different names to avoid conflicts with state keys
    graph.add_node("plan_node", plan_node)
    graph.add_node("search_node", search_node)
    graph.add_node("read_node", read_node)
    graph.add_node("reflect_node", reflect_node)
    graph.add_node("synthesize_node", synthesize_node)
    
    # Set entry point
    graph.set_entry_point("plan_node")
    
    # Add basic edges - simplified to avoid recursion issues
    graph.add_edge("plan_node", "search_node")
    graph.add_edge("search_node", "read_node")
    graph.add_edge("read_node", "reflect_node")
    
    # Simplified conditional edges based on LLM's own assessment
    graph.add_conditional_edges(
        "reflect_node",
        lambda s: has_sufficient_info(s),
        {
            True: "synthesize_node",
            False: "search_node"
        }
    )
    
    # Set finish point
    graph.set_finish_point("synthesize_node")
    
    return graph.compile()

# ── build the graph once at module load time -------------------------------
compiled_graph = build_graph()

# ── public API --------------------------------------------------------------
def answer(question: str) -> str:
    """
    Run the enhanced research loop to answer the question.
    
    Args:
        question: The user's question
        
    Returns:
        A comprehensive answer with research process details
    """
    logger.info(f"Starting research for question: {question[:100]}...")
    MEM.reset()
    MEM.add_query(question)
    
    try:
        final_state = compiled_graph.invoke({"question": question})
        result = final_state.get("answer") or "I don't know."
        
        logger.info(f"Research complete with {len(MEM.snips)} snippets and {len(MEM.reflections)} reflections")
        
        # Add memory summary with reflections and query evolution
        if MEM.reflections or len(MEM.queries) > 1:
            result += "\n\n" + MEM.summary()
        
        # Add sources
        from backend.agents.shared.publish import publish
        try:
            sources_text = publish(MEM.snips)
            result += "\n\n### Sources\n" + sources_text
        except Exception as e:
            logger.error(f"Error generating sources: {e}")
            result += f"\n\n### Sources\nError generating sources: {e}"
        
        return result
    except Exception as e:
        logger.error(f"Error in research process: {e}", exc_info=True)
        return f"Error during research: {str(e)}"
