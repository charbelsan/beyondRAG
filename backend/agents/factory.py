import yaml, pathlib
from typing import Dict, Any, Union, Optional

# Load configuration
CFG_PATH = pathlib.Path("configs/pipeline.yaml")
CFG: Dict[str, Any] = yaml.safe_load(CFG_PATH.read_text()) if CFG_PATH.exists() else {}

# Import agent implementations
from backend.agents.code_agent import answer as code_answer
from backend.agents.reasoning_agent import answer as reasoning_answer

def answer(query: str) -> str:
    """
    Factory function that routes queries to the appropriate agent based on configuration.
    
    Args:
        query: The user's query string
        
    Returns:
        The agent's response as a string
    """
    agent_mode = CFG.get("agent", {}).get("mode", "code")
    
    if agent_mode == "code":
        return code_answer(query)
    elif agent_mode == "reasoning":
        return reasoning_answer(query)
    elif agent_mode == "multi":
        # For multi-agent mode, we could implement a more sophisticated routing logic
        # For now, default to code agent
        return code_answer(query)
    else:
        return f"Error: Unknown agent mode '{agent_mode}'"
