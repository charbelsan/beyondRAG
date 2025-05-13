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
from smolagents import Tool

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
        
        # Aliases to prevent hallucinated tool calls from crashing
        "search": hybrid.hybrid_search,          # alias
        "wiki": hybrid.hybrid_search,            # crude but avoids crash
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
            # Use the tool_decorator to create a proper Tool instance
            decorated_func = tool_decorator(_auto_annotate(func))
            AGENT_TOOLS.append(decorated_func)
            logging.info(f"Added basic tool: {tool_name}")
        except Exception as e:
            logging.error(f"Error adding basic tool {tool_name}: {e}")

# Add research tools directly from RAW_TOOLS
research_tools_list = [
    "plan_research",
    "hybrid_search_with_content",
    "analyze_content",
    "reflect_on_research",
    "navigate_document_graph",
    "synthesize_answer"
]

for tool_name in research_tools_list:
    if tool_name in RAW_TOOLS:
        func = RAW_TOOLS[tool_name]
        try:
            # Create a Tool instance directly
            decorated_func = tool_decorator(_auto_annotate(func))
            AGENT_TOOLS.append(decorated_func)
            logging.info(f"Added research tool: {tool_name}")
        except Exception as e:
            logging.error(f"Error adding research tool {tool_name}: {e}")

# Log the tools that were successfully added
logging.info(f"Added {len(AGENT_TOOLS)} tools to the agent")

# ---------------------------------------------------------------------------
# Prompt template with enhanced research instructions
# ---------------------------------------------------------------------------
CODE_SYSTEM_PROMPT = """
You are Code‑Nav‑Agent, a Python‑capable assistant operating **inside a
restricted exec sandbox**. Always follow this protocol:

Thoughts:
- A short plain‑English reasoning about the next action. *One line.*

Code:
```py
# REQUIRED. Valid Python 3.10.
# Use ONLY the tools listed below (they are already imported):
#   bm25_search, vector_search, hybrid_search, read_span,
#   text_browser, walk_local, list_folder,
#   plan_research, hybrid_search_with_content, analyze_content,
#   reflect_on_research, navigate_document_graph, synthesize_answer
# Every tool call must be inside this code block.
# End the block with <end_code>
```<end_code>

Allowed imports: json, re, datetime, typing, numpy, math.

When your answer is complete:
Thoughts: FINISHED
Code:
```py
return "FINAL_ANSWER: <your_string_here>"
```<end_code>
"""

# Few-shot examples to lock the format
FEW_SHOT_EXAMPLES = """
User: What is the CLIP architecture?

Thoughts: I'll create a research plan first to understand CLIP architecture.

Code:
```py
plan = plan_research("What is the CLIP architecture?")
print(f"Research plan: {plan}")
```<end_code>

Execution logs:
Research plan: 1. Understand the basic concept of CLIP
2. Identify the key components of CLIP architecture
3. Explore how the image encoder works
4. Explore how the text encoder works
5. Understand how these components interact
6. Examine the training methodology
7. Identify the unique features of CLIP compared to other models

Thoughts: Now I'll search for relevant documents about CLIP.

Code:
```py
search_results = hybrid_search("CLIP architecture", k=5)
print(f"Found {len(search_results)} documents")

# Read the content of each document
for doc_id in search_results:
    content = read_span(doc_id, mode="auto", chars=750)
    print(f"Content from {doc_id}:\n{content[:100]}...")
```<end_code>

Execution logs:
Found 5 documents
Content from doc1: [clip.pdf p5 • 750c] Learning Transferable Visual Models From Natural Language Supervision...
Content from doc2: [clip.pdf p10 • 750c] The CLIP architecture consists of two main components: a vision encoder and a text encoder...

Thoughts: FINISHED

Code:
```py
return "FINAL_ANSWER: CLIP (Contrastive Language-Image Pre-training) is a neural network architecture that connects text and images. It consists of two main components: a vision encoder (based on a Vision Transformer or ResNet) and a text encoder (based on a Transformer). These encoders project images and text into a shared embedding space where related content is positioned closely together. CLIP is trained using contrastive learning on a large dataset of image-text pairs, allowing it to perform zero-shot classification and other cross-modal tasks."
```<end_code>
"""

# Create the CodeAgent with default prompt templates
AG = CodeAgent(
    model=LLM,
    tools=AGENT_TOOLS,
    # Don't use custom prompt templates, use the defaults
    # prompt_templates=PROMPTS,
    add_base_tools=False,
    max_steps=CONFIG.get("limits.max_steps_code_agent", 15),
    additional_authorized_imports=CONFIG.get("llm.authorized_imports", ["datetime", "re", "json", "typing", "numpy", "math"]),
)

# Add short-circuit parameters
AG.max_fails = 4  # after 4 parser errors, abort

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
    
    # Add safe globals for restricted_exec
    safe_globals = {"__builtins__": {}}

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

    # Add a reminder about citations if they're not already included
    if "Sources" not in final_answer_text and len(MEM.snips) > 0:
        final_answer_text += "\n\nRemember to include citations to your sources in your answer."

    return final_answer_text + "\n\n### Sources\n" + sources_text
