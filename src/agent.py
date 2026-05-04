import os
from typing import Annotated, TypedDict, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage

# Import the decomposer node from module_1
from module_1.decomposer import decomposer_node
from module_2.mapper import module_2_technical_node
from module_3.data_engineer import module_3_data_node
from module_4.ml_dev import module_4_ml_dev_node
from module_5.core_programmer import module_5_logic_dev_node
from module_6.integrator import module_6_integrator_node
from module_7.deployer import module_7_deployer_node

# Define the State for the LangGraph
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    blueprint: Dict[str, Any]
    is_complete: bool
    user_dataset_input: str   # Natural language from user about data

def ask_human(state: AgentState):
    """
    A dummy node that serves as a breakpoint for Human-in-the-Loop.
    LangGraph Studio will pause execution BEFORE running this node.
    The user can then input their natural language response.
    """
    pass

def ask_dataset(state: AgentState) -> dict:
    """
    Appends a question asking whether the user has an existing dataset.
    Graph pauses BEFORE receive_dataset so the user can type a path or 'no'.
    """
    import json
    telemetry = state.get("blueprint", {}).get("Technical_Mapping", {}).get("Telemetry_Variables", {})
    schema_str = json.dumps(telemetry, indent=2)
    return {
        "messages": [AIMessage(
            content=(
                "Technical mapping is complete.\n"
                f"Required RAN telemetry schema (FlexRIC-validated):\n{schema_str}\n\n"
                "Please specify your data availability. You can provide existing datasets or request synthetic generation.\n"
                "Examples:\n"
                "  • 'Generate all synthetic data' (or 'no')\n"
                "  • 'Use /path/to/data/ for all data'\n"
                "  • 'Use /path/to/ml_data/ for training/testing, but synthesize streaming data'\n\n"
                "Please describe what data you have and what needs to be synthesized. If you provide a path, we will profile it. If you need synthetic data, we will generate it."
            )
        )]
    }

def receive_dataset(state: AgentState) -> dict:
    """
    Interrupt node — graph pauses BEFORE this runs.
    After the user types their response, this node reads it and
    persists the dataset input into state.
    """
    last_msg = state["messages"][-1].content.strip() if state.get("messages") else "no"
    return {"user_dataset_input": last_msg}

def should_continue(state: AgentState):
    """
    After intent_decomposer runs, we ALWAYS pause for human input.
    If it's incomplete, we pause to ask for missing info.
    If it's complete, we pause to ask for the final 'CONFIRM'.
    """
    return "ask_human"

def check_confirmation(state: AgentState):
    """
    After the human provides input, check if the blueprint was already complete
    and if the human just confirmed it.
    """
    if not state["messages"]:
        return "intent_decomposer"

    last_msg = state["messages"][-1].content.strip().lower()

    # If the LLM previously marked it complete, and the human says 'confirm'
    if state.get("is_complete", False) and "confirm" in last_msg:
        return "technical_mapper"

    return "intent_decomposer"



def should_run_ml(state: AgentState):
    """
    Check if the ML node should run based on the cycle_Type.
    Blueprint sanitization for Pure_Logic happens in module_3_data_node before
    this edge is evaluated, so state is already clean here.
    """
    cycle_type = state.get("blueprint", {}).get("Intent_Blueprint", {}).get("cycle_Type", "Pure_Logic")
    if cycle_type in ["Supervised_ML", "Unsupervised_ML"]:
        return "ml_dev"
    return "logic_dev"

def ask_to_deploy(state: AgentState) -> dict:
    return {
        "messages": [AIMessage(
            content=(
                "Integration is complete and the final xApp is ready.\n"
                "Do you want to proceed with deploying to the testbed?\n"
                "Type 'Proceed' to deploy, or anything else to end."
            )
        )]
    }

def receive_deploy_decision(state: AgentState):
    """
    Interrupt node — graph pauses BEFORE this runs.
    """
    pass

def should_deploy(state: AgentState):
    if not state["messages"]:
        return END
    
    last_msg = state["messages"][-1].content.strip().lower()
    if "proceed" in last_msg:
        return "deployer"
    return END

# Initialize the StateGraph
builder = StateGraph(AgentState)

# Add Nodes
builder.add_node("intent_decomposer", decomposer_node)
builder.add_node("ask_human", ask_human)
builder.add_node("technical_mapper", module_2_technical_node)
builder.add_node("ask_dataset", ask_dataset)
builder.add_node("receive_dataset", receive_dataset)
builder.add_node("data_engineer", module_3_data_node)
builder.add_node("ml_dev", module_4_ml_dev_node)
builder.add_node("logic_dev", module_5_logic_dev_node)
builder.add_node("integrator", module_6_integrator_node)
builder.add_node("ask_to_deploy", ask_to_deploy)
builder.add_node("receive_deploy_decision", receive_deploy_decision)
builder.add_node("deployer", module_7_deployer_node)

# Set the entry point
builder.add_edge(START, "intent_decomposer")

# After intent_decomposer, always pause for human input
builder.add_conditional_edges(
    "intent_decomposer",
    should_continue,
    {
        "ask_human": "ask_human"
    }
)

# After ask_human, check if the human confirmed the complete blueprint
builder.add_conditional_edges(
    "ask_human",
    check_confirmation,
    {
        "technical_mapper": "technical_mapper",
        "intent_decomposer": "intent_decomposer"
    }
)

# After technical_mapper, ask the user about an existing dataset

builder.add_edge("technical_mapper", "ask_dataset")

builder.add_edge("ask_dataset", "receive_dataset")

# After receive_dataset, route to data_engineer
builder.add_edge("receive_dataset", "data_engineer")

# After data_engineer, check if ML is needed
builder.add_conditional_edges(
    "data_engineer",
    should_run_ml,
    {
        "ml_dev": "ml_dev",
        "logic_dev": "logic_dev"
    }
)

builder.add_edge("ml_dev", "logic_dev")
builder.add_edge("logic_dev", "integrator")
builder.add_edge("integrator", "ask_to_deploy")
builder.add_edge("ask_to_deploy", "receive_deploy_decision")

builder.add_conditional_edges(
    "receive_deploy_decision",
    should_deploy,
    {
        "deployer": "deployer",
        END: END
    }
)

builder.add_edge("deployer", END)

# Compile the graph with interrupts before both human-input nodes
graph = builder.compile(interrupt_before=["ask_human", "receive_dataset", "receive_deploy_decision"])

