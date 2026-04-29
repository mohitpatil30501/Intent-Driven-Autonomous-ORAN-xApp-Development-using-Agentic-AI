import os
from typing import Annotated, TypedDict, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Import the decomposer node from module_1
from module_1.decomposer import decomposer_node

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
        return END
        
    return "intent_decomposer"

# Initialize the StateGraph
builder = StateGraph(AgentState)

# Add Nodes
builder.add_node("intent_decomposer", decomposer_node)
builder.add_node("ask_human", ask_human)

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
        END: END,
        "intent_decomposer": "intent_decomposer"
    }
)

# Compile the graph with an interrupt before the ask_human node
graph = builder.compile(interrupt_before=["ask_human"])

