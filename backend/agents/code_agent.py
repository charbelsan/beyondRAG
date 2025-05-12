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

from smolagents import CodeAgent, OpenAIServerModel, PromptTemplates
from smolagents import tool as tool_decorator 
  


# ---------------------------------------------------------------------------
# Config and Shared Imports First
# ---------------------------------------------------------------------------
from backend.agents.shared.memory import MEM
from backend.agents.shared.publish import publish
from backend.agents.shared.reformulate import reformulate
from backend.config import CONFIG

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
            AGENT_TOOLS.append(tool_decorator(_auto_annotate(func)))
            logging.info(f"Added basic tool: {tool_name}")
        except Exception as e:
            logging.error(f"Error adding basic tool {tool_name}: {e}")

# Log the tools that were successfully added
logging.info(f"Added {len(AGENT_TOOLS)} tools to the agent")

# Note: The research tools from research_tools.py are already decorated with @tool
# and will be imported and used directly by the agent when needed

# ---------------------------------------------------------------------------
# Prompt template with enhanced research instructions
# ---------------------------------------------------------------------------
RESEARCH_SYSTEM_PROMPT = """
You are an expert research assistant for a private document corpus.

Follow this research loop:
1. PLAN: Create a research plan for answering the user's question using plan_research
2. SEARCH: Use hybrid_search_with_content to find relevant documents
3. READ: Analyze retrieved content using analyze_content to extract key information
4. REFLECT: Reflect on current knowledge using reflect_on_research and identify gaps
5. NAVIGATE: Use navigate_document_graph to explore related content when appropriate
6. SYNTHESIZE: Combine all gathered information into a comprehensive answer using synthesize_answer

During the REFLECT step, you should:
- Assess what you've learned so far
- Identify what information is still missing
- Determine the next research step
- If needed, reformulate your search query to better target missing information

Track your progress and reasoning throughout the research process.
When you have sufficient information, provide a final answer with citations.
"""

PROMPTS = PromptTemplates(
    system_prompt=CONFIG.get(
        "agent.system_prompt",
        RESEARCH_SYSTEM_PROMPT,
    )
)

# ---------------------------------------------------------------------------
# Build the CodeAgent (Keep as is)
# ---------------------------------------------------------------------------
AG = CodeAgent(
    model=LLM,
    tools=AGENT_TOOLS,
    #prompt_templates=PROMPTS,
    add_base_tools=False,
    max_steps=CONFIG.get("limits.max_steps", 40),
    additional_authorized_imports=CONFIG.get("llm.authorized_imports", ["datetime"]),
)

# ---------------------------------------------------------------------------
# Public function the rest of the pipeline imports (Keep as is)
# ---------------------------------------------------------------------------
def answer(q: str) -> str:
    """
    Reformulate query, reset shared memory, run CodeAgent once, stitch in
    memory summary & sources block. Uses RAW_TOOLS for execution.
    """
    q = reformulate(q)
    MEM.reset()
    MEM.add_query(q)

    # --- Run Agent ---
    try:
        # Assuming AG.run handles the loop internally based on max_steps
        response = AG.run(q) # Pass only the initial query
    except Exception as agent_err:
        logging.error(f"Agent run failed: {agent_err}", exc_info=True)
        # Consider returning a more user-friendly error or re-raising
        return f"Agent execution failed: {agent_err}"

    # --- Post-processing ---
    # Check if the response is just the final answer text or includes the marker
    if response.startswith("FINAL_ANSWER:"):
         final_answer_text = response.split("FINAL_ANSWER:", 1)[1].strip()
    else:
         # Assume the agent directly returned the answer text if no marker
         final_answer_text = response

    # Append memory summary if needed
    if MEM.reflections or len(MEM.queries) > 1:
        final_answer_text += "\n\n" + MEM.summary()

    # Append sources (MEM should have been populated during AG.run's internal tool calls)
    # Ensure publish function handles potential errors gracefully
    try:
        sources_text = publish(MEM.snips)
    except Exception as pub_err:
        logging.error(f"Publishing sources failed: {pub_err}", exc_info=True)
        sources_text = "Error generating sources."

    return final_answer_text + "\n\n### Sources\n" + sources_text

