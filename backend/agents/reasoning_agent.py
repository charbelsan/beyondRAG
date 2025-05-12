"""
backend/agents/reasoning_agent.py
---------------------------------
LangGraph‑based reasoning agent with PDF‑aware retrieval.
"""

from __future__ import annotations

import os
from typing import List, Optional

from langgraph.graph import StateGraph
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from backend.tools.bm25 import bm25_search
from backend.tools.vector import vector_search
from backend.tools.read_span import read_span
from backend.agents.shared.memory import MEM
from backend.config import CONFIG

# ── LLM ---------------------------------------------------------------------
LLM = ChatOpenAI(
    base_url=os.getenv("OPENAI_API_BASE", CONFIG.get("llm.base_url", "http://localhost:11434/v1")),
    api_key =os.getenv("OPENAI_API_KEY",  CONFIG.get("llm.api_key",  "ollama")),
    model_name=os.getenv("OPENAI_MODEL_NAME", CONFIG.get("llm.model_id", "qwen2.5:32b-instruct")),
    temperature=0.2,
)

# ── state schema ------------------------------------------------------------
class QAState(BaseModel):
    question: str
    context:  List[str] = []
    answer:   Optional[str] = None

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

# ── node: retrieval ---------------------------------------------------------
def retrieve_node(state: QAState) -> dict:
    """Hybrid search; fallback to raw_text if read_span is empty."""
    query = state.question

    bm25_ids   = _extract_ids(bm25_search(query, k=5))
    vector_ids = _extract_ids(vector_search(query, k=5))
    top_ids    = list(dict.fromkeys(bm25_ids + vector_ids))[:5]

    snippets = []
    from pathlib import Path
    import json
    meta = json.loads(Path(CONFIG.get_path("indexing.meta")).read_text())

    for did in top_ids:
        txt = read_span(did, mode="auto", chars=750)
        if not txt:                       # fallback to stored raw_text
            txt = meta.get(did, {}).get("raw_text", "")
        if txt:
            snippets.append(txt)
            MEM.track(txt, did, mode="auto")

    return {"context": snippets}

# ── node: synthesis ---------------------------------------------------------
_SYST_PROMPT = (
    "You are an expert research assistant. "
    "Answer the user’s question using the context passages below. "
    "If the context is insufficient, say you don’t know."
)
_SYNTH_TEMPLATE = PromptTemplate(
    template="{system_prompt}\n\nQuestion:\n{question}\n\nContext:\n{context}\n\nAnswer:",
    input_variables=["system_prompt", "question", "context"],
)

def synthesise_node(state: QAState) -> dict:
    ctx = "\n\n".join(state.context) or "[no useful context]"
    llm_input = _SYNTH_TEMPLATE.format(
        system_prompt=_SYST_PROMPT,
        question=state.question,
        context=ctx,
    )
    answer = LLM.invoke(llm_input).content.strip()
    return {"answer": answer}

# ── build graph -------------------------------------------------------------
graph = StateGraph(state_schema=QAState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("synthesise", synthesise_node)
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "synthesise")
graph.set_finish_point("synthesise")
compiled_graph = graph.compile()

# ── public API --------------------------------------------------------------
def answer(question: str) -> str:
    MEM.reset()
    final_state = compiled_graph.invoke({"question": question})
    result = final_state.get("answer") or "I don't know."
    if MEM.reflections:
        result += "\n\n" + MEM.summary()
    return result
