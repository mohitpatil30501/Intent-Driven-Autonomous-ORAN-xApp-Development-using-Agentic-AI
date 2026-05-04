import os
import requests
from langchain_core.tools import tool

SEMANTIC_SEARCH_URL = os.getenv("SEMANTIC_SEARCH_URL", "http://localhost:7080")

@tool
def semantic_search_summary(query: str, n_results: int = 5) -> str:
    """
    Search the codebase for semantic meaning (e.g., 'how to update throughput').
    Returns full code snippets to provide complete context.
    Use this to get a broad overview of where concepts are implemented.
    """
    try:
        res = requests.post(
            f"{SEMANTIC_SEARCH_URL}/semantic_search",
            json={
                "query": query, 
                "n_results": n_results, 
                "truncate_chars": 0,
                "return_full_text": True
            },
            timeout=30
        )
        if res.status_code == 200:
            return res.json().get("results", "No results found.")
        return f"Error: Status code {res.status_code}"
    except Exception as e:
        return f"Error connecting to semantic search: {e}"

@tool
def semantic_search_detailed(query: str, n_results: int = 2) -> str:
    """
    Search the codebase for semantic meaning and return the FULL code bodies.
    WARNING: This consumes a lot of tokens. Use ONLY when you specifically need
    to see the complete implementation details of a function you found using
    semantic_search_summary. Limit n_results to 1 or 2.
    """
    try:
        res = requests.post(
            f"{SEMANTIC_SEARCH_URL}/semantic_search",
            json={
                "query": query, 
                "n_results": n_results, 
                "truncate_chars": 0,
                "return_full_text": True
            },
            timeout=30
        )
        if res.status_code == 200:
            return res.json().get("results", "No results found.")
        return f"Error: Status code {res.status_code}"
    except Exception as e:
        return f"Error connecting to semantic search: {e}"
