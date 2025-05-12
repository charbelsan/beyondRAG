# File: backend/agents/code_agent.py
# ============================
# CodeAgent wired to R.A.G. project tools.
# Uses explicit imports and delayed tool list creation
# to avoid circular import issues.
# Assumes tool functions are decorated with @tool in their respective files.
# ============================

from __future__ import annotations

import inspect

import os
import ast
import logging
import time
from types import MappingProxyType
from typing import Any, Callable, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("code_agent")

from smolagents import CodeAgent, OpenAIServerModel, PromptTemplates
from smolagents import tool as tool_decorator 
  


# ---------------------------------------------------------------------------
# Config and Shared Imports First
# ---------------------------------------------------------------------------
from backend.agents.shared.memory import MEM
from backend.agents.shared.publish import publish
from backend.agents.shared.reformulate import reformulate
from backend.config import CONFIG

# Add tracking for research stages
class ResearchStage:
    PLAN = "PLAN"
    SEARCH = "SEARCH"
    READ = "READ"
    REFLECT = "REFLECT"
    NAVIGATE = "NAVIGATE"
    SYNTHESIZE = "SYNTHESIZE"

# ---------------------------------------------------------------------------
# Explicitly import each tool module
# Let Python fully load each one before we try to access their contents.
# ---------------------------------------------------------------------------
try:
    from backend.tools import bm25
    from backend.tools import vector
    from backend.tools import hybrid
    from backend.tools import read_span
    from backend.tools import local_browser
    from backend.tools import walk_local
    from backend.tools import list_folder
    from backend.tools import research_tools  # Import the new research tools
    from backend.tools import restricted_exec # Import separately if needed
    ALL_TOOL_MODULES_IMPORTED = True
except ImportError as e:
    logging.error(f"Failed to import tool modules: {e}", exc_info=True)
    # Handle the error appropriately - maybe raise it or exit
    raise ImportError(f"Could not import all tool modules: {e}") from e
    ALL_TOOL_MODULES_IMPORTED = False


# ---------------------------------------------------------------------------
# LLM Initialization (Keep as is)
# ---------------------------------------------------------------------------
LLM = OpenAIServerModel(
    model_id=os.getenv("OPENAI_MODEL_NAME",
                       CONFIG.get("llm.model_id", "mistral")),
    api_base=os.getenv("OPENAI_API_BASE",
                       CONFIG.get("llm.base_url", "http://localhost:11434")),
    api_key=os.getenv("OPENAI_API_KEY") or CONFIG.get("llm.api_key") or "local-key",
)

# ---------------------------------------------------------------------------
# Tool Definitions - ***Define AFTER imports***
# ---------------------------------------------------------------------------
# Check if imports succeeded before defining tools
if not ALL_TOOL_MODULES_IMPORTED:
    raise RuntimeError("Tool module imports failed, cannot define tools.")

# Agent Tools - List the decorated functions directly using the imported modules
# (Assumes @tool decorator is applied in bm25.py, vector.py, etc.)
# AGENT_TOOLS: List[Callable] = [
#     bm25.bm25_search,
#     vector.vector_search,
#     hybrid.hybrid_search,
#     read_span.read_span,
#     local_browser.text_browser, # Ensure this function exists and is decorated
#     walk_local.walk_local,
#     list_folder.list_folder,
# ]



# Raw callables map for restricted_exec
RAW_TOOLS: Dict[str, Callable] = MappingProxyType(
    {
        # Basic search and navigation tools
        "bm25_search":   bm25.bm25_search,
        "vector_search": vector.vector_search,
        "hybrid_search": hybrid.hybrid_search,
        "read_span":     read_span.read_span,
        "text_browser":  local_browser.text_browser,
        "walk_local":    walk_local.walk_local,
        "list_folder":   list_folder.list_folder,
        
        # Advanced research tools
        "plan_research": research_tools.plan_research,
        "hybrid_search_with_content": research_tools.hybrid_search_with_content,
        "analyze_content": research_tools.analyze_content,
        "reflect_on_research": research_tools.reflect_on_research,
        "navigate_document_graph": research_tools.navigate_document_graph,
        "synthesize_answer": research_tools.synthesize_answer,
    }
)

def _auto_annotate(fn: Callable) -> Callable:
    """
    Ensure every parameter (and return) in *fn* has a type‑hint.
    Missing hints are filled with `typing.Any`, then the function is returned.
    """
    sig = inspect.signature(fn)
    ann = dict(fn.__annotations__)  # copy existing
    for name in sig.parameters:
        ann.setdefault(name, Any)
    ann.setdefault("return", Any)
    fn.__annotations__ = ann
    return fn

# Create a list of all tools with proper annotations
AGENT_TOOLS = []

# First add the basic tools that we know work
basic_tools = [
    "bm25_search",
    "vector_search",
    "hybrid_search",
    "read_span",
    "text_browser",
    "walk_local",
    "list_folder"
]

for tool_name in basic_tools:
    if tool_name in RAW_TOOLS:
        func = RAW_TOOLS[tool_name]
        try:
            # Use the function directly without re-decorating it
            AGENT_TOOLS.append(_auto_annotate(func))
            logging.info(f"Added basic tool: {tool_name}")
        except Exception as e:
            logging.error(f"Error adding basic tool {tool_name}: {e}")

# Add research tools - these are already decorated with @tool
# We need to import them directly rather than trying to re-decorate them
from backend.tools.research_tools import (
    plan_research,
    hybrid_search_with_content,
    analyze_content,
    reflect_on_research,
    navigate_document_graph,
    synthesize_answer
)

# Add the research tools directly
research_tools_list = [
    plan_research,
    hybrid_search_with_content,
    analyze_content,
    reflect_on_research,
    navigate_document_graph,
    synthesize_answer
]

for func in research_tools_list:
    try:
        AGENT_TOOLS.append(_auto_annotate(func))
        logging.info(f"Added research tool: {func.__name__}")
    except Exception as e:
        logging.error(f"Error adding research tool {getattr(func, '__name__', 'unknown')}: {e}")

# Log the tools that were successfully added
logging.info(f"Added {len(AGENT_TOOLS)} tools to the agent")

# ---------------------------------------------------------------------------
# Prompt template with enhanced research instructions
# ---------------------------------------------------------------------------
RESEARCH_SYSTEM_PROMPT = """
You are an expert research assistant for a private document corpus.

Follow this research loop EXACTLY in order:
1. PLAN: First, create a research plan using plan_research
2. SEARCH: Use hybrid_search_with_content to find relevant documents
3. READ: Analyze retrieved content using analyze_content
4. REFLECT: Use reflect_on_research to assess what you've learned and what's missing
5. If reflect_on_research indicates sufficient_info is False, go back to SEARCH with the reformulated query
6. SYNTHESIZE: When sufficient_info is True, use synthesize_answer to create the final answer

IMPORTANT RULES:
1. Always track your progress through these steps explicitly.
2. For each step, state which step you're on and what you're doing.
3. Stop after 5 iterations to prevent infinite loops.
4. Always check the return values of tools for errors.
5. When using hybrid_search_with_content, check if new_docs_found is 0 and try a different query if so.
6. When using reflect_on_research, always check the sufficient_info flag to decide whether to continue searching.

Your final answer should be comprehensive, well-structured, and include citations to the sources you used.
"""

PROMPTS = PromptTemplates(
    system_prompt=CONFIG.get(
        "agent.system_prompt",
        RESEARCH_SYSTEM_PROMPT,
    ),
    managed_agent="""
    You are a research assistant with access to tools for searching and analyzing documents.
    Follow the research loop: PLAN → SEARCH → READ → REFLECT → SYNTHESIZE.
    Use the available tools to find information and answer the user's question.
    """,
    final_answer="""
    Based on your research, provide a comprehensive answer to the user's question.
    Include citations to the sources you used.
    """,
    planning="""
    Create a plan to answer the user's question using the available tools.
    Break down the task into steps and explain your approach.
    """
)

# ---------------------------------------------------------------------------
# Build the CodeAgent (Keep as is)
# ---------------------------------------------------------------------------
AG = CodeAgent(
    model=LLM,
    tools=AGENT_TOOLS,
    prompt_templates=PROMPTS,
    add_base_tools=False,
    max_steps=CONFIG.get("limits.max_steps", 40),
    additional_authorized_imports=CONFIG.get("llm.authorized_imports", ["datetime", "re"]),
)

# ---------------------------------------------------------------------------
# Public function the rest of the pipeline imports (Keep as is)
# ---------------------------------------------------------------------------
def answer(q: str) -> str:
    """
    Reformulate query, reset shared memory, run CodeAgent once, stitch in
    memory summary & sources block. Uses RAW_TOOLS for execution.
    """
    logger.info(f"Starting code agent research for query: {q[:100]}...")
    
    reformulated_q = reformulate(q)
    if reformulated_q != q:
        logger.info(f"Query reformulated to: {reformulated_q[:100]}...")
    else:
        reformulated_q = q
    
    MEM.reset()
    MEM.add_query(reformulated_q)
    
    # Add tracking for research stages
    if not hasattr(MEM, 'stages'):
        MEM.stages = []
    
    # Add method to track stages if it doesn't exist
    if not hasattr(MEM, 'add_stage'):
        def add_stage(stage, content):
            MEM.stages.append((stage, content))
            logger.info(f"Research stage: {stage} - {content[:50]}...")
        MEM.add_stage = add_stage
    
    # Initialize stages
    MEM.stages = []
    MEM.add_stage(ResearchStage.PLAN, "Starting research process")
    
    # Log available tools
    logger.info(f"Available tools: {', '.join(RAW_TOOLS.keys())}")
    logger.info(f"Agent configured with {len(AGENT_TOOLS)} tools")

    # --- Run Agent ---
    try:
        # Add a tool usage tracker
        original_run = AG.run
        
        def run_with_logging(query):
            logger.info(f"Starting agent execution with query: {query[:100]}...")
            start_time = time.time()
            result = original_run(query)
            duration = time.time() - start_time
            logger.info(f"Agent execution completed in {duration:.2f} seconds")
            return result
            
        AG.run = run_with_logging
        
        # Assuming AG.run handles the loop internally based on max_steps
        response = AG.run(reformulated_q) # Pass only the initial query
        
        logger.info(f"Agent completed with {len(MEM.snips)} snippets and {len(MEM.reflections)} reflections")
        
        # Log research stages
        if hasattr(MEM, 'stages') and MEM.stages:
            logger.info(f"Research stages: {len(MEM.stages)}")
            for i, (stage, content) in enumerate(MEM.stages):
                logger.info(f"Stage {i+1}: {stage}")
        
    except Exception as agent_err:
        logger.error(f"Agent run failed: {agent_err}", exc_info=True)
        # Consider returning a more user-friendly error or re-raising
        return f"Agent execution failed: {agent_err}"

    # --- Post-processing ---
    # Check if the response is just the final answer text or includes the marker
    if response.startswith("FINAL_ANSWER:"):
         final_answer_text = response.split("FINAL_ANSWER:", 1)[1].strip()
         logger.info("Found FINAL_ANSWER marker in response")
    else:
         # Assume the agent directly returned the answer text if no marker
         final_answer_text = response
         logger.info("No FINAL_ANSWER marker found, using full response")

    # Append memory summary if needed
    if MEM.reflections or len(MEM.queries) > 1:
        logger.info(f"Adding memory summary with {len(MEM.reflections)} reflections and {len(MEM.queries)} queries")
        summary = MEM.summary()
        
        # Add research stages to summary if available
        if hasattr(MEM, 'stages') and MEM.stages:
            stages_summary = "\n\n## Research Process\n"
            for i, (stage, _) in enumerate(MEM.stages):
                stages_summary += f"{i+1}. {stage}\n"
            summary += stages_summary
            
        final_answer_text += "\n\n" + summary

    # Append sources (MEM should have been populated during AG.run's internal tool calls)
    # Ensure publish function handles potential errors gracefully
    try:
        sources_text = publish(MEM.snips)
        logger.info(f"Added {len(MEM.snips)} sources to response")
    except Exception as pub_err:
        logger.error(f"Publishing sources failed: {pub_err}", exc_info=True)
        sources_text = "Error generating sources."

    return final_answer_text + "\n\n### Sources\n" + sources_text

