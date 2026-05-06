import os
import sys
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama

# Add src to path for tool imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.deployer.testbed.testbed_tool import deploy_xapp_to_testbed
from tools.workspace.workspace_tools import workspace_tools

MODULE_7_SYSTEM_PROMPT = """You are "Module 7: The xApp Deployer".
Your job is to deploy the final xApp and any associated artifacts (like ML models) to the testbed.

WORKFLOW:
1. Read the `final_xapp.py` file from the workspace.
2. Check the Blueprint for any ML model artifacts (e.g., in `ml/` directory).
3. If artifacts exist, read them. For binary files like `.pkl`, you may need to use a tool that returns base64 or just pass the filename if the tool supports it.
4. Call the `deploy_xapp_to_testbed` tool with the `xapp_code` and an `artifacts` dictionary (filename -> content).
5. Provide a summary of the logs returned by the tool.

CRITICAL: Do NOT attempt to manually copy files or run docker commands. Use the tool provided.
"""

def get_llm():
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1")
    return ChatOllama(model=ollama_model, base_url=ollama_url)

def module_7_deployer_node(state: dict) -> dict:
    """Module 7: Deploys the xApp using the testbed tool."""
    blueprint = state.get("blueprint", {})
    
    llm = get_llm()
    # Combine workspace tools (for reading final_xapp.py) and the testbed tool
    module_7_tools = workspace_tools + [deploy_xapp_to_testbed]
    
    deployer_agent = create_react_agent(
        model=llm,
        tools=module_7_tools,
        prompt=MODULE_7_SYSTEM_PROMPT
    )
    
    prompt_content = "Please read `final_xapp.py` and deploy it to the testbed."
    
    try:
        result = deployer_agent.invoke(
            {"messages": [HumanMessage(content=prompt_content)]}
        )
        final_text = result["messages"][-1].content
    except Exception as e:
        error_msg = f"Module 7 Error (Deployment): {str(e)}"
        print(error_msg)
        return {"messages": [AIMessage(content=error_msg)]}

    return {
        "blueprint": blueprint,
        "messages": [AIMessage(content=f"Module 7 Deployment Result:\n{final_text}")],
        "is_complete": True
    }
