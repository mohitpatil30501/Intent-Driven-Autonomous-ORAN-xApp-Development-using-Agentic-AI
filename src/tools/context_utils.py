from langchain_core.messages import ToolMessage, AIMessage

def limit_tool_messages(agent_state: dict) -> dict:
    """
    State modifier to prevent context window explosion from large tool outputs.
    Truncates all ToolMessages that have already been 'used' by the LLM
    (i.e., they appear before an AIMessage).
    Also truncates any 'active' ToolMessages if they are excessively large.
    """
    messages = agent_state.get("messages", [])
    if not messages:
        return {}
        
    filtered = []
    
    # We want to identify the 'active' tool messages, which are the ones 
    # that haven't been responded to by an AIMessage yet.
    # To do this safely, we scan backwards. As long as we see ToolMessages,
    # they are part of the active turn. Once we hit an AIMessage, the active turn ends.
    
    active_tool_ids = set()
    hit_ai_message = False
    
    for m in reversed(messages):
        if getattr(m, "type", "") == "ai" or isinstance(m, AIMessage):
            hit_ai_message = True
            
        is_tool = isinstance(m, ToolMessage) or getattr(m, "type", "") == "tool"
        if is_tool and not hit_ai_message:
            active_tool_ids.add(getattr(m, "tool_call_id", None))
            
    # Now build the filtered list forward
    for m in messages:
        is_tool = isinstance(m, ToolMessage) or getattr(m, "type", "") == "tool"
        if is_tool:
            tool_call_id = getattr(m, "tool_call_id", None)
            content = m.content
            
            if tool_call_id in active_tool_ids:
                # Active tool message. Truncate if it's too large (e.g., > 10000 chars)
                if isinstance(content, str) and len(content) > 10000:
                    content = content[:10000] + "\n\n... [Output truncated due to size limit]"
                    kwargs = {
                        "content": content,
                        "tool_call_id": tool_call_id,
                        "name": m.name,
                        "status": getattr(m, "status", "success")
                    }
                    if hasattr(m, "id") and m.id:
                        kwargs["id"] = m.id
                    filtered.append(ToolMessage(**kwargs))
                else:
                    filtered.append(m)
            else:
                # Used tool message. Truncate entirely.
                kwargs = {
                    "content": "[Output truncated to maintain context window.]",
                    "tool_call_id": tool_call_id,
                    "name": m.name,
                    "status": getattr(m, "status", "success")
                }
                if hasattr(m, "id") and m.id:
                    kwargs["id"] = m.id
                filtered.append(ToolMessage(**kwargs))
        else:
            filtered.append(m)
            
    return {"messages": filtered}

def limit_context_window(agent_state: dict, max_messages: int = 14) -> dict:
    """
    Combines tool message truncation with a sliding window for the conversation.
    Keeps the first message (usually the task) and the N most recent messages.
    """
    # First, truncate large/old tool outputs
    limited_state = limit_tool_messages(agent_state)
    messages = limited_state.get("messages", [])
    
    if len(messages) <= max_messages:
        return {"messages": messages}
    
    # Keep the first message (System or initial Human) and the last N-1 messages
    return {"messages": [messages[0]] + messages[-(max_messages-1):]}
