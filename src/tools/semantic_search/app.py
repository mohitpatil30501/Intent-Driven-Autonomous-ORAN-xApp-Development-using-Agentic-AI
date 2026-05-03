import os
import yaml
import subprocess
import threading
from git import Repo
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

app = FastAPI()

# Config
REPO_DIR = "/app/cloned_repos"
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))

# Initialize Chroma and Embeddings
chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
collection = chroma_client.get_or_create_collection(name="oran_codebase")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # Fast, runs on CPU easily

def log_progress(msg: str):
    print(msg, flush=True)
    with open("/app/ingestion.log", "a") as f:
        f.write(msg + "\n")

def clone_or_pull_repos():
    if not os.path.exists(REPO_DIR):
        os.makedirs(REPO_DIR)
        
    with open("/app/repos.yml", "r") as f:
        config = yaml.safe_load(f)
        
    for repo_conf in config.get("repositories", []):
        target_path = os.path.join(REPO_DIR, repo_conf["name"])
        if os.path.exists(target_path):
            log_progress(f"Pulling latest for {repo_conf['name']}...")
            repo = Repo(target_path)
            repo.remotes.origin.pull()
        else:
            log_progress(f"Cloning {repo_conf['name']}...")
            Repo.clone_from(repo_conf["url"], target_path, branch=repo_conf.get("branch", "master"))

def ingest_code_to_vector_db():
    log_progress("Chunking code... (Loading files and parsing syntax)")
    # Load C, C++, and Python files
    loader = GenericLoader.from_filesystem(
        REPO_DIR,
        glob="**/*",
        suffixes=[".c", ".h", ".cpp", ".hpp", ".py"],
        parser=LanguageParser()
    )
    docs = loader.load()
    
    # Split code intelligently based on functions/classes
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.C, chunk_size=1000, chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)
    
    # Format for Chroma
    documents = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    total = len(documents)
    log_progress(f"Total chunks to embed: {total}")
    
    # Embed and upload (in batches to save memory)
    batch_size = 100
    for i in range(0, total, batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_vecs = embeddings.embed_documents(batch_docs)
        collection.upsert(
            documents=batch_docs,
            embeddings=batch_vecs,
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
        log_progress(f"Embedded {min(i+batch_size, total)} / {total} chunks...")
    log_progress("Ingestion Complete!")

is_ready = False

def background_ingestion():
    global is_ready
    try:
        with open("/app/ingestion.log", "w") as f:
            f.write("System Startup Initialized...\n")
        clone_or_pull_repos()
        ingest_code_to_vector_db()
    except Exception as e:
        log_progress(f"Error during ingestion: {e}")
    finally:
        is_ready = True

@app.get("/status")
def get_status():
    """Check the status of the ingestion process."""
    try:
        with open("/app/ingestion.log", "r") as f:
            logs = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        logs = ["Log file not created yet."]
    
    return {
        "is_ready": is_ready,
        "logs": logs
    }

# Run clone and ingest on startup in the background
@app.on_event("startup")
def startup_event():
    threading.Thread(target=background_ingestion, daemon=True).start()

# --- ENDPOINTS FOR YOUR LANGGRAPH AGENT ---

class QueryReq(BaseModel):
    query: str
    n_results: int = 5
    truncate_chars: int = 800
    return_full_text: bool = False

@app.post("/semantic_search")
def semantic_search(req: QueryReq):
    """Find code based on meaning (e.g. 'throughput calculation')"""
    if not is_ready:
        return {"results": "System Notice: The search engine is currently building the AI Knowledge Graph (cloning and embedding repositories). This takes a few minutes on the first run. Please try your request again shortly."}
        
    query_embedding = embeddings.embed_query(req.query)
    results = collection.query(query_embeddings=[query_embedding], n_results=req.n_results)
    
    formatted = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        source = meta.get('source', 'Unknown File')
        content = doc
        if not req.return_full_text and req.truncate_chars > 0 and len(content) > req.truncate_chars:
            content = content[:req.truncate_chars] + "\n... [TRUNCATED to save tokens. Set return_full_text=True for full code]"
        formatted.append(f"--- File: {source} ---\n{content}\n")
    return {"results": "\n".join(formatted)}

@app.post("/exact_search")
def exact_search(req: QueryReq):
    """Find exact keyword matches using ripgrep (e.g. 'dl_aggr_tbs')"""
    if not is_ready:
        return {"results": "System Notice: The search engine is currently cloning repositories. Please try your request again shortly."}
        
    try:
        # Runs: rg -n "query" /app/cloned_repos
        result = subprocess.run(
            ["rg", "-n", req.query, REPO_DIR], 
            capture_output=True, text=True, timeout=10
        )
        out = result.stdout.split('\n')[:req.n_results*2] # limit output lines
        return {"results": "\n".join(out) if out else "No matches found."}
    except Exception as e:
        return {"error": str(e)}