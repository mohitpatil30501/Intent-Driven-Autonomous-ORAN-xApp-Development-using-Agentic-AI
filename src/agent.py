import os
from typing import Annotated, TypedDict, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage

# Import the decomposer node from module_1
from module_1.decomposer import decomposer_node
from module_2.mapper import module_2_technical_node
from module_3.synthesizer import module_3_data_node
from module_3.profiler import dataset_profiler_node
from module_4.ml_dev import module_4_ml_dev_node
from module_5.core_programmer import module_5_logic_dev_node
from module_6.integrator import module_6_integrator_node

# Define the State for the LangGraph
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    blueprint: Dict[str, Any]
    is_complete: bool
    user_dataset_path: Optional[str]   # None = auto-synthesize; str = user-provided path

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
    telemetry = state.get("blueprint", {}).get("Technical_Mapping", {}).get("Telemetry_Variables", [])
    cols = [v.get("C_variable", "?") for v in telemetry]
    return {
        "messages": [AIMessage(
            content=(
                "Technical mapping is complete.\n"
                f"Required RAN telemetry columns (FlexRIC-validated): {cols}\n\n"
                "Do you have an existing dataset you would like to use?\n"
                "  • Type 'no' to auto-generate synthetic data\n"
                "  • Or paste an absolute path to your data file or directory\n"
                "    (e.g., /home/user/spotlight_dataset/ or /data/traffic.csv)\n\n"
                "Multi-file and nested-folder datasets are supported. The system will "
                "discover files, filter to RAN-reportable columns only (verified against "
                "the FlexRIC codebase), and map them to the required telemetry variables."
            )
        )]
    }

def receive_dataset(state: AgentState) -> dict:
    """
    Interrupt node — graph pauses BEFORE this runs.
    After the user types their response, this node reads it and
    persists the dataset path into state (or None for 'no').
    """
    last_msg = state["messages"][-1].content.strip() if state.get("messages") else "no"
    if last_msg.lower() in ("no", "n", "synthesize", ""):
        return {"user_dataset_path": None}
    return {"user_dataset_path": last_msg}

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

def check_dataset_input(state: AgentState) -> str:
    """
    Conditional edge (read-only): routes to dataset_profiler if the user
    provided a path, otherwise routes to the existing data_synthesizer.
    """
    path = state.get("user_dataset_path")
    if path and path.strip():
        return "dataset_profiler"
    return "data_synthesizer"

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

# Initialize the StateGraph
builder = StateGraph(AgentState)

# Add Nodes
builder.add_node("intent_decomposer", decomposer_node)
builder.add_node("ask_human", ask_human)
builder.add_node("technical_mapper", module_2_technical_node)
builder.add_node("ask_dataset", ask_dataset)
builder.add_node("receive_dataset", receive_dataset)
builder.add_node("data_synthesizer", module_3_data_node)
builder.add_node("dataset_profiler", dataset_profiler_node)
builder.add_node("ml_dev", module_4_ml_dev_node)
builder.add_node("logic_dev", module_5_logic_dev_node)
builder.add_node("integrator", module_6_integrator_node)

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

# After receive_dataset, route to profiler or synthesizer
builder.add_conditional_edges(
    "receive_dataset",
    check_dataset_input,
    {
        "data_synthesizer": "data_synthesizer",
        "dataset_profiler": "dataset_profiler",
    }
)

# After data_synthesizer, check if ML is needed
builder.add_conditional_edges(
    "data_synthesizer",
    should_run_ml,
    {
        "ml_dev": "ml_dev",
        "logic_dev": "logic_dev"
    }
)

# After dataset_profiler, same ML routing gate
builder.add_conditional_edges(
    "dataset_profiler",
    should_run_ml,
    {
        "ml_dev": "ml_dev",
        "logic_dev": "logic_dev",
    }
)

builder.add_edge("ml_dev", "logic_dev")
builder.add_edge("logic_dev", "integrator")
builder.add_edge("integrator", END)

# Compile the graph with interrupts before both human-input nodes
graph = builder.compile(interrupt_before=["ask_human", "receive_dataset"])
