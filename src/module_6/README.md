Module 6: xApp Synthesizer (The Integrator)

What it does: Takes the validated XAppLogic from Module 5 and injects it into the FlexRIC Python SDK template. Maps the FlexRIC C-structs to the inputs of process_interval().

Key design decisions:

- Only the fields Module 6 needs are extracted from the full blueprint (Technical_Mapping, Logic_Artifacts, and a brief Intent_Blueprint summary). ML artifacts, data paths, and synthesizer metadata are excluded to keep the initial prompt small.
- A pre_model_hook trims the agent's internal message list before each model call: it keeps the first message (the task) plus the 10 most recent, dropping the middle. This prevents context-window overflow on OSS models regardless of how many iterations occur.
- The system prompt enforces exactly 4 tool calls: (1) flexric_rag_context RAG lookup, (2) read flexric_template.py, (3) write final_xapp.py, (4) one combined terminal command for the syntax check and log. This eliminates the open-ended loop that caused the agent to exhaust its recursion budget.
- Default recursion limit is 20 (down from 40). Set INTEGRATOR_RECURSIVE_LIMIT to override.
- If the recursion limit is hit, the node returns a clear error message with the env var to raise, rather than propagating the opaque LangGraph "need more steps" message.
