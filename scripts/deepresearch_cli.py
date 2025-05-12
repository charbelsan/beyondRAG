#!/usr/bin/env python
"""
deepresearch_cli.py  –  quick terminal interface to the DeepResearch agent


"""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
import argparse
import sys
import logging # Added for potential error logging


# Import the central answer function from the agent factory
from backend.agents.factory import answer

def main() -> None:
    """
    Parses command-line arguments and runs the DeepResearch agent either
    in one-shot mode or interactive REPL mode.
    """
    parser = argparse.ArgumentParser(
        description="Ask DeepResearch from the command line. Uses the agent configured in pipeline.yaml."
    )
    parser.add_argument(
        "query",
        nargs="*", # Allows zero or more arguments for the query
        help="Your question. If omitted, enters interactive REPL.",
    )
    # Optional: Add arguments to override config settings, e.g., --agent-mode
    # parser.add_argument("--agent-mode", choices=['code', 'reasoning'], help="Override agent mode from config")

    args = parser.parse_args()

    # Optional: Handle config overrides
    # if args.agent_mode:
    #     from backend.config import CONFIG
    #     print(f"[CLI] Overriding agent mode to: {args.agent_mode}")
    #     CONFIG.config['agent']['mode'] = args.agent_mode # Example override

    if args.query:
        # --- One-shot mode ---
        full_query = " ".join(args.query)
        print(f"[Query] {full_query}")
        print("-" * 60)
        try:
            response = answer(full_query)
            print(response)
        except Exception as e:
            logging.exception("Error processing one-shot query:") # Log full traceback
            print(f"\n[Error] {e}")
            sys.exit(1) # Exit with error code
        return

    # --- Interactive REPL mode ---
    print("Entering interactive mode. Type 'exit', 'quit', or 'q' to leave.")
    try:
        while True:
            # Use input() to get user query
            try:
                 q = input("❯ ").strip()
            except UnicodeDecodeError:
                 print("[Error] Invalid input encoding. Please use UTF-8.", file=sys.stderr)
                 continue


            if q.lower() in {"exit", "quit", "q"}:
                print("Exiting.")
                break
            if q:
                print("-" * 60)
                try:
                    response = answer(q)
                    print(response)
                except Exception as e:
                     logging.exception("Error processing interactive query:") # Log full traceback
                     print(f"\n[Error] {e}")
                print("-" * 60)
    except (EOFError, KeyboardInterrupt):
        # Handle Ctrl+D (EOFError) or Ctrl+C (KeyboardInterrupt) gracefully
        print("\nExiting.")

if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
    main()
