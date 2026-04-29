import os
import json
import re
import unittest
from typing import Dict, Any

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from agent import graph
from module_1.decomposer import get_llm, extract_json

INTENTS = [
    "I want an xApp to monitor per-UE throughput and adjust PRBs if it drops below 5 Mbps. This is to avoid slice starvation. I think pure logic is fine.",
    "Build an xApp to block users who are consuming too much bandwidth.",
    "We need to predict network congestion using machine learning and proactively steer traffic.",
    "Create an unsupervised ML model to detect anomalies in base station buffer occupancy and restart the gNB when an anomaly occurs.",
    "I need to ensure URLLC slice meets 1ms latency. If latency exceeds 1ms, increase priority."
]

USER_SIMULATOR_PROMPT = """You are a network operator requesting an automated application (xApp).
Your overall goal is: {intent}

The assistant is trying to help you formalize this into a blueprint. 
They have asked you some questions to clarify your requirements.
Answer their questions directly and naturally. Provide realistic technical details if necessary.
Make up any missing details logically (e.g. streaming data rates, historical data size, exact metrics).
Keep your response concise.
"""

JUDGE_PROMPT = """You are an expert evaluator. Evaluate the extracted JSON Blueprint for an xApp intent.
Original Intent from user: {intent}

Extracted JSON Blueprint:
{blueprint}

Rubric (0 to 5 points):
- Does it capture the target action correctly? (1 pt)
- Does it capture the objective/why correctly? (1 pt)
- Are the requested telemetry metrics appropriate for the goal? (1 pt)
- Is the cycle type appropriately selected? (1 pt)
- Are the data requirements well-defined based on the cycle type? (1 pt)

Return ONLY a JSON block containing your evaluation with this exact structure:
```json
{{
  "score": <int>,
  "reasoning": "<string>"
}}
```
"""

class TestModule1AgenticFlow(unittest.TestCase):
    def setUp(self):
        self.llm = get_llm()

    def simulate_user_response(self, original_intent: str, conversation_history: list) -> str:
        # Construct the user simulator messages
        sys_msg = SystemMessage(content=USER_SIMULATOR_PROMPT.format(intent=original_intent))
        # Pass the history so the simulator knows what the assistant asked
        messages = [sys_msg] + conversation_history
        response = self.llm.invoke(messages)
        return response.content

    def evaluate_blueprint(self, original_intent: str, blueprint: dict) -> dict:
        prompt = JUDGE_PROMPT.format(intent=original_intent, blueprint=json.dumps(blueprint, indent=2))
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        # Extract json from response
        try:
            return extract_json(response.content)
        except Exception as e:
            # Fallback regex search if extract_json fails
            match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
            return {"score": 0, "reasoning": f"Failed to parse judge output. Error: {e}"}

    def test_end_to_end_intents(self):
        results = []
        
        for idx, intent in enumerate(INTENTS):
            print(f"\n--- Testing Intent {idx+1} ---")
            print(f"Initial Intent: {intent}")
            
            # Initialize State
            state = {"messages": [HumanMessage(content=intent)]}
            
            max_turns = 5
            turns = 0
            
            while turns < max_turns:
                turns += 1
                
                # Invoke the intent decomposer graph
                # The graph will run the decomposer node and return the updated state
                new_state = graph.invoke(state)
                
                # Update our local state tracker
                state["messages"] = new_state["messages"]
                
                is_complete = new_state.get("is_complete", False)
                
                if is_complete:
                    print(f"Blueprint completed in {turns} turns.")
                    break
                    
                # If not complete, we need the simulated user to answer the assistant's questions
                assistant_msg = state["messages"][-1] # The last message from the assistant
                
                print(f"[Assistant query]: {assistant_msg.content[:150]}...")
                
                user_reply = self.simulate_user_response(intent, state["messages"])
                print(f"[Simulated User]: {user_reply[:150]}...")
                
                # Add the user's reply to the state
                state["messages"].append(HumanMessage(content=user_reply))
                
            blueprint = new_state.get("blueprint", {})
            
            # LLM-as-a-judge Evaluation
            eval_result = self.evaluate_blueprint(intent, blueprint)
            score = eval_result.get("score", 0)
            reasoning = eval_result.get("reasoning", "No reasoning provided.")
            
            print(f"Final Score: {score}/5")
            print(f"Reasoning: {reasoning}")
            
            results.append({"intent": intent, "score": score, "turns": turns, "completed": is_complete})
            
        print("\n--- Summary ---")
        for r in results:
            print(f"Score {r['score']}/5 | Turns {r['turns']} | Complete: {r['completed']} | Intent: {r['intent'][:50]}...")

        successful = sum(1 for r in results if r['score'] >= 4)
        print(f"Total Successful: {successful}/{len(INTENTS)}")

if __name__ == "__main__":
    unittest.main()
