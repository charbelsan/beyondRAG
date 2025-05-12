"""
pipelines/deepresearch_pipeline.py
──────────────────────────────────
• Conforms to the Open-WebUI *Pipelines* loader:
      class Pipeline → class Valves → pipe()
• Reads LLM connection details from configs/pipeline.yaml via ConfigManager
• Sets environment variables for LLM connections
• Optional AUTO_INDEX valve — rebuilds Whoosh + FAISS on container start
• May also be launched *stand-alone* (without Pipelines)  
  by running:  `python -m pipelines.deepresearch_pipeline --port 9123`
"""

from __future__ import annotations

import os, json, pathlib, asyncio, sys, argparse
from typing import Dict, Any, Union, Generator, Iterator

from pydantic import BaseModel, Field
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# Import configuration manager
from backend.config import CONFIG

# Import agent factory and indexing builder
from backend.agents.factory import answer
from indexing import builder

# import logging, sys
# logging.basicConfig(
#     stream=sys.stdout,
#     level=logging.INFO,          # or DEBUG for chatty output
#     format="%(asctime)s %(levelname)s %(name)s: %(message)s",
#     force=True,                  # override other configs
# )


# Get system prompt from configuration
_SYSTEM_PROMPT = CONFIG.get("agent.system_prompt", "")

# ╭──────────────────────────────────────────────────────────────╮
# │  Pipeline class recognized by Open WebUI                     │
# ╰──────────────────────────────────────────────────────────────╯
class Pipeline:
    class Valves(BaseModel):
        MODEL_ID: str 
        AUTO_INDEX: bool 

    def __init__(self) -> None:
        """Initialize the pipeline with default valves."""
        self.valves = self.Valves(
            **{
                "MODEL_ID": os.getenv("MODEL_ID", "deepresearch"),
                "AUTO_INDEX": os.getenv("AUTO_INDEX", "false"),
                
            }
        )

    async def on_startup(self) -> None:
        """
        Startup hook called by Open WebUI when the container starts.
        
        This method:
        1. Rebuilds indexes if AUTO_INDEX is enabled
        """
        if self.valves.AUTO_INDEX:
            try:
                # Ensure indexing directories exist
                CONFIG.ensure_dir("indexing.root")
                CONFIG.ensure_dir("indexing.whoosh")
                
                # Rebuild indexes
                builder.rebuild()
                print("[Pipeline] 👍 Index rebuilt")
            except Exception as exc:
                print(f"[Pipeline] ⚠️ Index rebuild failed: {exc}", file=sys.stderr)

    def pipe(self, body: Dict[str, Any]) -> Union[str, Generator, Iterator]:
        """
        Main entry point for each OpenAI chat request.
        
        Args:
            body: The request body from the OpenAI chat API
            
        Returns:
            The response from the agent
        """
        # Extract user message
        user_msg: str = body["messages"][-1]["content"]
        
        # Prepend system prompt if available
        full_prompt = f"{_SYSTEM_PROMPT}\n\n{user_msg}" if _SYSTEM_PROMPT else user_msg
        
        # Process with the appropriate agent
        return answer(full_prompt)

# ╭──────────────────────────────────────────────────────────────╮
# │  Optional: run this file directly to expose a local server   │
# ╰──────────────────────────────────────────────────────────────╯
def _create_app() -> FastAPI:
    """
    Create a FastAPI application for standalone operation.
    
    Returns:
        A configured FastAPI application
    """
    # Create pipeline instance
    pipe = Pipeline()
    
    # Create FastAPI app
    app = FastAPI(
        title="DeepResearch-Standalone",
        description="Local server for DeepResearch Pipeline",
        version="1.0.0"
    )

    @app.on_event("startup")
    async def _startup():
        """Run the same initialization hook as in the pipeline."""
        await pipe.on_startup()

    @app.post("/v1/chat/completions")
    async def completions(payload: Dict[str, Any]):
        """
        Minimal OpenAI-schema wrapper for chat completions.
        
        Args:
            payload: The request payload
            
        Returns:
            JSONResponse with OpenAI-compatible format
        """
        # Process the request
        result = pipe.pipe(payload)
        
        # Convert generators/iterators to string
        if isinstance(result, (Generator, Iterator)):
            result = "".join(result)
            
        # Return in OpenAI format
        return JSONResponse({
            "id": "deepresearch-local",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result},
                "finish_reason": "stop"
            }],
            "model": CONFIG.get("llm.model_id", "local-model")
        })

    return app

if __name__ == "__main__":
    """Run the standalone server when this file is executed directly."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run DeepResearch pipeline as a local server")
    parser.add_argument("--port", type=int, default=9123, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--log-level", type=str, default="info", help="Logging level")
    args = parser.parse_args()
    
    # Print startup message
    print(f" DeepResearch standalone server starting on http://{args.host}:{args.port}")
    
    # Run the server
    uvicorn.run(
        _create_app(), 
        host=args.host, 
        port=args.port, 
        log_level=args.log_level
    )
