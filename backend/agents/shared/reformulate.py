# backend/agents/shared/reformulate.py
"""
reformulate.py
──────────────
The *query-reformulation* step is now handled **implicitly** inside the agent’s
planning loop (the model is free to change the query it passes to any search
tool).  Therefore this helper becomes a thin no-op shim, kept only so that all
agent code can continue to call `reformulate(query)` safely.

Nothing here touches `ChatOpenAI`, so there is **no dependency on an
`OPENAI_API_KEY`** and no risk of import-time failures.
"""


def reformulate(query: str) -> str:
    """
    Backwards-compatible stub.

    Parameters
    ----------
    query : str
        The original user query.

    Returns
    -------
    str
        Exactly the same query, unchanged.
    """
    return query
