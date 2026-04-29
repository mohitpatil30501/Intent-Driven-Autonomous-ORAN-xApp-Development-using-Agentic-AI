# Agent Chat WebUI

This directory contains the LangChain `agent-chat-ui` frontend, pre-configured to run locally via Docker Compose.

> **Credits**: The original web application design and core codebase were developed by [LangChain AI (agent-chat-ui)](https://github.com/langchain-ai/agent-chat-ui).

## Configuration

Before starting the WebUI, ensure your `.env` file in this directory is correctly configured:

```env
# LangGraph API Server URL (Required)
NEXT_PUBLIC_API_URL=http://localhost:2024

# Assistant ID (Required)
NEXT_PUBLIC_ASSISTANT_ID=agent

# Optional. Required for Agent Builder deployments.
# Set to "langsmith-api-key" when using a LangSmith PAT.
NEXT_PUBLIC_AUTH_SCHEME=

# LangSmith API Key (Required if connecting to deployed LangGraph servers)
LANGSMITH_API_KEY=
```

## Running Locally

To build and start the WebUI container, run:

```bash
docker compose up -d --build
```

### Accessing the WebUI

Once running, the application will be available at:
👉 **[http://localhost:3000](http://localhost:3000)**

## Stopping

To stop the WebUI container:

```bash
docker compose down
```

## Viewing Logs

If you need to view the server logs or debug Next.js output:

```bash
docker compose logs -f
```

## Customization & Development
The provided `docker-compose.yml` mounts the current directory as a volume inside the container. This means you can freely edit the code locally in your IDE, and Next.js will automatically hot-reload your changes in the browser without needing a container restart.
