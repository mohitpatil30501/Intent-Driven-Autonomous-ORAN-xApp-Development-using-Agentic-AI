# LangGraph Source Code

This directory contains the source code for the LangGraph agent implementation, specifically the `agent.py` file which defines the LangGraph agent.

## Configuration

Before running the agent, ensure your `.env` file in this directory is correctly configured. This file is used to inject environment variables into the LangGraph execution.

```env
# LangGraph API Server URL (Required for running the agent)
NEXT_PUBLIC_API_URL=http://localhost:2024

# Assistant ID (Required for running the agent)
NEXT_PUBLIC_ASSISTANT_ID=agent

# Optional. Required for Agent Builder deployments.
# Set to "langsmith-api-key" when using a LangSmith PAT.
NEXT_PUBLIC_AUTH_SCHEME=

# LangSmith API Key (Required if connecting to deployed LangGraph servers)
LANGSMITH_API_KEY=
```

## Running the Agent

The LangGraph agent is typically run using the LangChain CLI. Start the development server without automatic reloading:

```bash
# Start the LangGraph development server
langgraph dev --no-reload
```

## Viewing Logs

To view the server logs or debug LangChain output:

```bash
langgraph logs
```