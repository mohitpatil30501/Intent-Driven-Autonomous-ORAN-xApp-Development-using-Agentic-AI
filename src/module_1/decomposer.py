import json
import re
import os
import copy
from typing import Any, Dict
from langchain_core.messages import SystemMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain_core.runnables.config import RunnableConfig
from dotenv import load_dotenv

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.context_utils import limit_context_window

load_dotenv("src/.env")

DECOMPOSER_PROMPT = """You are "Module 1: The Intent Decomposer" in an automated network application (xApp) development pipeline.
Your ONLY job is to interview the human user, understand their high-level business requirement, and extract that intent into a strict JSON Blueprint.

CRITICAL RULE: DO NOT HALLUCINATE TECHNICAL SPECIFICATIONS. 
You do not know about O-RAN, FlexRIC, C-structs, or specific API variables. Your job is purely to capture the human's requirements in plain Natural Language (NL). Module 2 will handle the technical database mapping later.

--- HOW TO HANDLE INCOMPLETE REQUESTS (HUMAN-IN-THE-LOOP) ---
To build this application, you MUST know all of the following:
1. WHAT metrics the xApp needs to monitor (e.g., throughput, buffer occupancy).
2. WHAT action the xApp should take when its condition is met (e.g., change bandwidth, block user).
3. WHY it is doing this (the objective/goal).
4. THE CYCLE TYPE (Pure_Logic, Supervised_ML, or Unsupervised_ML).
5. THE DATA BEHAVIOR (If ML is used, what should the historical training data look like? What anomalies or patterns should we simulate?).
6. THE MODEL ACCEPTANCE CRITERIA (ONLY if ML is used). If the human does not specify a threshold, use threshold 0.85 and metric_policy "task_aware". For Pure_Logic, ALL three model_acceptance_criteria fields MUST be set to null.

If the user's request is missing any required intent, action, objective, cycle type, telemetry, or data behavior details, set "isComplete" to false, list the missing fields, and politely ask the user targeted questions to fill in the blanks. DO NOT GUESS.
For ML acceptance criteria only, do NOT block completion when the human has not specified a threshold; apply the default threshold 0.85 and metric_policy "task_aware".
IMPORTANT: If cycle_Type is Pure_Logic, set threshold, metric_policy, and metric_description_NL all to null. Do NOT output numeric threshold values for Pure_Logic xApps.

--- RESPONSE FORMAT ---
You MUST output your response in exactly TWO sections:
1. A conversational message addressing the human. (If incomplete, ask your questions. If complete, summarize the plan and ask them to type 'CONFIRM').
2. A strict JSON code block containing the blueprint state.

--- JSON BLUEPRINT TEMPLATE ---
```json
{
  "Intent_Blueprint": {
    "validation": {
      "isComplete": false,
      "missingFields": ["list of fields you need the human to clarify"],
      "questionsForHuman": ["The exact questions you are asking the user"]
    },
    "xApp_Name": "string (A clean, snake_case name for the project)",
    "code_language": "string (The programming language to be used for the xApp) [C++, Python3]",
    "objective_Why": "string (Why do we need this? e.g., 'To prevent slice starvation' or 'TBD')",
    "target_Action_What_NL": "string (What action will it take? e.g., 'Modify PRB allocation' or 'TBD')",
    "cycle_Type": "Enum: [Supervised_ML, Unsupervised_ML, Pure_Logic, TBD]",
    "Service Models": "List of Service Models (ORAN Specific e.g.: Slice Service Model, MAC SM, GTP, etc)",
    "requested_Telemetry_NL": [
      "List of metrics needed in plain English, e.g., 'per slice throughput' or 'TBD'"
    ],
    "data_Requirements": {
      "needs_historical_training_data": false,
      "historical_data_description_NL": "string (If ML is chosen, describe what the training data should look like, e.g., 'Needs 10,000 rows showing normal traffic and network congestion spikes' or null)",
      "streaming_mock_data_description_NL": "string (Describe what the real-time data stream should look like to trigger the logic/model)"
    },
    "model_acceptance_criteria": {
      "threshold": "number (ONLY for Supervised_ML or Unsupervised_ML. Set to null for Pure_Logic.)",
      "metric_policy": "string (ONLY for Supervised_ML or Unsupervised_ML. Set to null for Pure_Logic.)",
      "metric_description_NL": "string (ONLY for ML types. For Pure_Logic, this MUST be null.)"
    }
  }
}
```
"""

def get_llm():
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1")
    return ChatOllama(model=ollama_model, base_url=ollama_url)

def extract_json(text: str) -> Dict[str, Any]:
    # Extract json block from markdown
    json_match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    return {}

def decomposer_node(state: dict, config: RunnableConfig) -> dict:
    llm = get_llm()
    messages = state.get("messages", [])
    
    # Prepend the system prompt if not present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=DECOMPOSER_PROMPT)] + messages
        
    # Limit context window to prevent slowdowns
    limited_state = limit_context_window({"messages": messages}, max_messages=14)
    messages = limited_state["messages"]

    response = llm.invoke(messages, config)
    
    # Parse out the JSON
    parsed_json = extract_json(response.content)
    
    is_complete = False
    new_blueprint = copy.deepcopy(state.get("blueprint", {})) if isinstance(state.get("blueprint"), dict) else {}
    
    if parsed_json:
        is_complete = parsed_json.get("Intent_Blueprint", {}).get("validation", {}).get("isComplete", False)
        if "Intent_Blueprint" in parsed_json:
            new_blueprint["Intent_Blueprint"] = parsed_json["Intent_Blueprint"]
        
    return {
        "messages": [response],
        "blueprint": new_blueprint,
        "is_complete": is_complete
    }
