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
        "bm25_search":   bm25.bm25_search,
        "vector_search": vector.vector_search,
        "hybrid_search": hybrid.hybrid_search,
        "read_span":     read_span.read_span,
        "text_browser":  local_browser.text_browser,
        "walk_local":    walk_local.walk_local,
        "list_folder":   list_folder.list_folder,
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

AGENT_TOOLS = [
    tool_decorator(_auto_annotate(func))   # decorator → Tool instance
    for func in RAW_TOOLS.values()
]

# ---------------------------------------------------------------------------
# Prompt template (Keep as is)
# ---------------------------------------------------------------------------
PROMPTS = PromptTemplates(
    system_prompt=CONFIG.get(
        "agent.system_prompt",
        "You are an expert research assistant for a private corpus.",
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

