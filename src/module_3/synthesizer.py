import json
import re
import os
import sys
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama

# Add the src folder to path so we can import from tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.workspace.workspace_tools import workspace_tools

MODULE_3_SYSTEM_PROMPT = """You are "Module 3: The O-RAN Data Engineer" in an automated xApp development pipeline.
Your job is to generate synthetic datasets based on the provided Blueprint and Technical Mapping.

--- STRICT WORKFLOW INSTRUCTIONS ---
You must use your provided tools to execute the following steps in order:

1. CREATE DIRECTORY: Create a folder named `data/` inside the workspace.
2. WRITE SCRIPT: Write a Python script (e.g., `data/generate_data.py`) using `pandas` and `numpy`.
   - Look at `Technical_Mapping.Telemetry_Variables`. The `C_variable` values MUST be the exact column headers in your generated CSV files.
   - Look at `Intent_Blueprint.data_Requirements` for the math/logic needed to simulate anomalies or traffic spikes.
3. EXECUTE SCRIPT: Run the script using your terminal_command tool (e.g., `python data/generate_data.py`).
4. CROSS-CHECK (MANDATORY): You MUST verify the data. Run a command like `head -n 6 data/streaming_mock_data.csv` to view the headers and the first 5 rows. 
   - Verify that the headers exactly match the required C-struct variables.
   - Verify that the data values make mathematical sense based on the requirements.
   - If the data is wrong, rewrite the script and run it again.

--- WHAT TO GENERATE ---
Check `Intent_Blueprint.cycle_Type`. 
If it is "Pure_Logic":
- Generate ONLY `data/streaming_mock_data.csv` (approx 100-500 rows).

If it is "Supervised_ML" or "Unsupervised_ML":
- Generate `data/streaming_mock_data.csv` (approx 100-500 rows).
- AND Generate `data/historical_training_data.csv` (approx 5,000+ rows) based on the `historical_data_description_NL`. Include a 'label' column if Supervised_ML.

--- RESPONSE FORMAT ---
ONLY after you have successfully verified the first 5 rows, output a final JSON block updating the blueprint with the paths to the generated data.

```json
{
  "Data_Paths": {
    "streaming_mock_data_path": "data/streaming_mock_data.csv",
    "historical_training_data_path": "data/historical_training_data.csv" // or null if Pure_Logic
  }
}
```
"""

def get_llm():
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1")
    return ChatOllama(model=ollama_model, base_url=ollama_url)

RECURSION_LIMIT = 8

def module_3_data_node(state: dict) -> dict:
    """Module 3: Synthesizes and verifies CSV mock data."""
    
    blueprint = state.get("blueprint", {})
    
    prompt_content = (
        f"Here is the complete Blueprint and Technical Mapping:\n"
        f"{json.dumps(blueprint, indent=2)}\n\n"
        f"Create the `data/` directory, write the python script inside it, generate the CSVs, "
        f"verify the first 5 rows using the terminal, and return the Data_Paths JSON."
    )
    
    llm = get_llm()
    
    # Create the ReAct agent
    data_agent = create_react_agent(
        model=llm, 
        tools=workspace_tools, 
        prompt=MODULE_3_SYSTEM_PROMPT
    )
    
    try:
        # Recursion limit increased to 8 to account for:
        # 1. mkdir -> 2. write file -> 3. run python -> 4. run head -n 6 -> 5. output JSON
        # Plus a few extra steps if it makes a syntax error and needs to retry.
        result = data_agent.invoke(
            {"messages": [HumanMessage(content=prompt_content)]},
            {"recursion_limit": RECURSION_LIMIT}
        )
        final_text = result["messages"][-1].content
        
    except Exception as e:
        print(f"Module 3 Error (Data Gen/Verify): {e}")
        return {"messages": [AIMessage(content=f"Data generation failed: {e}")]}

    # Extract the JSON block
    data_paths = {}
    json_match = re.search(r'```json\s*(.*?)\s*```', final_text, re.DOTALL)
    if json_match:
        try:
            data_paths = json.loads(json_match.group(1)).get("Data_Paths", {})
            blueprint["Data_Paths"] = data_paths
        except json.JSONDecodeError:
            print("Failed to parse JSON from Module 3 output.")
            pass

    return {
        "blueprint": blueprint,
        "messages": [AIMessage(content=final_text)]
    }
