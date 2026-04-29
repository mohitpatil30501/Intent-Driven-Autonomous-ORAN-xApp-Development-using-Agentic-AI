# Agent Tools

This directory contains containerized tools utilized by the agent. Each tool has a `docker-compose.yml` for easy deployment.

## 1. OrioSearch
A web search and content extraction API. This connects to your centralized LLM via the top-level `.env` file.
- **Start**: `cd oriosearch && docker compose up -d`
- **Stop**: `cd oriosearch && docker compose down`
- **URL**: `http://localhost:8000`

## 2. Semantic Search
A local, AI-native code search engine powered by ChromaDB. It automatically clones targeted repositories, embeds the codebase, and exposes an API for semantic and exact keyword search, replacing the need for standalone GitLab and Sourcegraph instances.
- **Start**: `cd semantic_search && docker compose up -d --build`
- **Stop**: `cd semantic_search && docker compose down`
- **URL**: `http://localhost:7080`
- **Documentation**: See [semantic_search/README.md](semantic_search/README.md) for detailed configuration (like defining target repositories in `repos.yml`).