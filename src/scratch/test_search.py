import os
import requests
import json

SEMANTIC_SEARCH_URL = "http://localhost:7080"

def test_search(query):
    print(f"Testing search for: {query}")
    try:
        res = requests.post(
            f"{SEMANTIC_SEARCH_URL}/semantic_search",
            json={
                "query": query, 
                "n_results": 5, 
                "truncate_chars": 0,
                "return_full_text": True
            },
            timeout=30
        )
        if res.status_code == 200:
            results = res.json().get("results", [])
            print(f"Found {len(results)} results")
            for i, r in enumerate(results):
                print(f"\n--- Result {i+1} ---")
                # Print first 200 chars and last 200 chars to see if it's full
                content = r
                if len(content) > 1000:
                    print(content[:500] + "\n...\n" + content[-500:])
                else:
                    print(content)
        else:
            print(f"Error: {res.status_code}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_search("SLICE SM indication message structure")
    test_search("fr_slice_t struct definition")
