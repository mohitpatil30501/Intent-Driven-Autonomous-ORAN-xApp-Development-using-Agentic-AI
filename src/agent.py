import os
from typing import Annotated, TypedDict, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Import the decomposer node from module_1
from module_1.decomposer import decomposer_node
from module_2.mapper import module_2_technical_node
from module_3.synthesizer import module_3_data_node
from module_4.ml_dev import module_4_ml_dev_node
from module_5.core_programmer import module_5_logic_dev_node
from module_6.integrator import module_6_integrator_node

# Define the State for the LangGraph
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    blueprint: Dict[str, Any]
    is_complete: bool

def ask_human(state: AgentState):
    """
    A dummy node that serves as a breakpoint for Human-in-the-Loop.
    LangGraph Studio will pause execution BEFORE running this node.
    The user can then input their natural language response.
    """
    pass

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
    """
    cycle_type = state.get("blueprint", {}).get("Intent_Blueprint", {}).get("cycle_Type", "Pure_Logic")
    if cycle_type in ["Supervised_ML", "Unsupervised_ML"]:
        return "ml_dev"
    return "logic_dev"

# Initialize the StateGraph
builder = StateGraph(AgentState)

# Add Nodes
builder.add_node("intent_decomposer", decomposer_node)
builder.add_node("ask_human", ask_human)
builder.add_node("technical_mapper", module_2_technical_node)
builder.add_node("data_synthesizer", module_3_data_node)
builder.add_node("ml_dev", module_4_ml_dev_node)
builder.add_node("logic_dev", module_5_logic_dev_node)
builder.add_node("integrator", module_6_integrator_node)

# Set the entry point
builder.add_edge(START, "intent_decomposer")

# Add conditional edges from the intent_decomposer
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

builder.add_edge("technical_mapper", "data_synthesizer")

# After data_synthesizer, check if ML is needed
builder.add_conditional_edges(
    "data_synthesizer",
    should_run_ml,
    {
        "ml_dev": "ml_dev",
        "logic_dev": "logic_dev"
    }
)

builder.add_edge("ml_dev", "logic_dev")
builder.add_edge("logic_dev", "integrator")
builder.add_edge("integrator", END)

# Compile the graph with an interrupt before the ask_human node
graph = builder.compile(interrupt_before=["ask_human"])

