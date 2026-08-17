"""
AI Agents Microservice  —  /agents/*
Separate pod from the main AI-NOC gateway.
Exposes a single /agents/chat endpoint that orchestrates all specialist agents.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from orchestrator import detect_agent, model_for_agent, model_name_for_agent
from agents import research_agent, bioinfo_agent


UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/data/research-uploads"))
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024 * 1024  # 20 GB Cap

ALLOWED_EXTENSIONS = {
    # Documents
    "pdf", "docx", "txt", "md",
    # Data Tables
    "csv", "xlsx", "tsv",
    # Genomic & Bioinfo
    "fasta", "fa", "fna", "fastq", "fq", "vcf", "gff", "gff3", "bed",
    # Protein & Molecular Structures
    "pdb", "mol", "mol2", "sdf", "smi", "smiles",
    # Images
    "png", "jpg", "jpeg", "tiff", "tif"
}

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure upload directory exists when the container boots up
    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Storage directory verified at: {UPLOAD_DIR.resolve()}")
    except Exception as e:
        logger.error(f"❌ Failed to create UPLOAD_DIR: {e}")

    logger.info("Agents microservice starting up")
    yield
    logger.info("Agents microservice shutting down")


app = FastAPI(
    title="AI Scientific Agents",
    description="Multi-agent scientific platform — Research, Bioinfo, Chemistry, Coding, Statistics",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ───────────────────────────────────────────────

class ChatTurn(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str


class AgentChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    history: list[ChatTurn] = Field(default_factory=list)
    agent_override: str | None = Field(
        default=None,
        description="Force a specific agent: research|bioinfo|chemistry|coding|statistics|general",
    )


class AgentChatResponse(BaseModel):
    answer: str
    agent: str
    model: str
    sources: list[dict] = []
    context_hits: dict = {}
    detected_agent: str


class FileMetadata(BaseModel):
    filename: str
    size_bytes: int
    size_human: str
    extension: str
    category: str


class FileListResponse(BaseModel):
    files: List[FileMetadata]
    total_files: int
    total_size_human: str


# ── Helper Functions ──────────────────────────────────────────────────────────

def _human_readable_size(size_bytes: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def _categorize_file(ext: str) -> str:
    ext = ext.lower()
    if ext in ["pdf", "docx", "txt", "md"]:
        return "Document"
    if ext in ["csv", "xlsx", "tsv"]:
        return "Tabular Data"
    if ext in ["fasta", "fa", "fna", "fastq", "fq", "vcf", "gff", "gff3", "bed"]:
        return "Bioinformatics"
    if ext in ["pdb", "mol", "mol2", "sdf", "smi", "smiles"]:
        return "Chemistry / Structure"
    if ext in ["png", "jpg", "jpeg", "tiff", "tif"]:
        return "Image"
    return "Other"


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "agents-service running", "version": "1.0.0"}


@app.get("/agents/status")
def agent_status():
    """List available agents and their backing models."""
    return {
        "agents": {
            "research":   {"model": model_name_for_agent("research"),   "status": "active"},
            "bioinfo":    {"model": model_name_for_agent("bioinfo"),    "status": "active"},
            "chemistry":  {"model": model_name_for_agent("chemistry"),  "status": "coming_soon"},
            "coding":     {"model": model_name_for_agent("coding"),     "status": "coming_soon"},
            "statistics": {"model": model_name_for_agent("statistics"), "status": "coming_soon"},
            "general":    {"model": model_name_for_agent("general"),    "status": "active"},
        }
    }


@app.post("/agents/chat", response_model=AgentChatResponse)
async def agent_chat(req: AgentChatRequest):
    """
    Main agent endpoint.
    1. Detect intent → select agent
    2. Dispatch to specialist agent
    3. Return grounded answer with sources
    """
    history = [t.model_dump() for t in req.history]

    # Detect or override agent
    detected = detect_agent(req.message, history)
    agent = req.agent_override if req.agent_override else detected

    model_url  = model_for_agent(agent)
    model_name = model_name_for_agent(agent)

    logger.info("Agent=%s model=%s message='%s...'", agent, model_name, req.message[:80])

    # Dispatch to appropriate agent execution layer
    try:
        if agent == "research":
            result = await research_agent.run(req.message, history, model_url, model_name)
        elif agent == "bioinfo":
            result = await bioinfo_agent.run(req.message, history, model_url, model_name)
        # elif agent == "chemistry":
        #     result = await chemistry_agent.run(req.message, history, model_url, model_name)
        # elif agent == "coding":
        #     result = await coding_agent.run(req.message, history, model_url, model_name)
        # elif agent == "statistics":
        #     result = await statistics_agent.run(req.message, history, model_url, model_name)
        else:
            # General fallback — direct LLM call, no RAG
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{model_url}/v1/chat/completions",
                    json={
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": "You are a helpful scientific assistant."},
                            *history[-6:],
                            {"role": "user", "content": req.message},
                        ],
                        "max_tokens": 1024,
                        "temperature": 0.7,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                answer = resp.json()["choices"][0]["message"]["content"]
            result = {"answer": answer, "sources": [], "agent": "general", "model": model_name, "context_hits": {}}

    except httpx.HTTPStatusError as e:
        logger.error("LLM backend error: %s", e)
        raise HTTPException(status_code=502, detail=f"LLM backend returned {e.response.status_code}")
    except Exception as e:
        logger.error("Agent execution error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return AgentChatResponse(
        answer=result["answer"],
        agent=result.get("agent", agent),
        model=result.get("model", model_name),
        sources=result.get("sources", []),
        context_hits=result.get("context_hits", {}),
        detected_agent=detected,
    )


# ── Upload Endpoints ──────────────────────────────────────────────────────────

@app.get("/uploads", response_model=FileListResponse)
def list_uploads():
    """Returns metadata for all research files stored on the node path."""
    if not UPLOAD_DIR.exists():
        return FileListResponse(files=[], total_files=0, total_size_human="0 B")

    file_list = []
    total_bytes = 0

    for file_path in UPLOAD_DIR.glob("*"):
        if file_path.is_file():
            size = file_path.stat().st_size
            total_bytes += size
            ext = file_path.suffix.lstrip(".").lower()
            
            file_list.append(
                FileMetadata(
                    filename=file_path.name,
                    size_bytes=size,
                    size_human=_human_readable_size(size),
                    extension=ext,
                    category=_categorize_file(ext)
                )
            )

    return FileListResponse(
        files=file_list,
        total_files=len(file_list),
        total_size_human=_human_readable_size(total_bytes)
    )


@app.post("/uploads")
async def upload_file(file: UploadFile = File(...)):
    """
    Streams file uploads directly to node path.
    Supports large datasets (up to 20 GB) via chunked writes.
    """
    filename = Path(file.filename).name  # Sanitize basic filename
    ext = filename.split(".")[-1].lower() if "." in filename else ""

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type extension '.{ext}'. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # Ensure directory exists before writing
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    target_path = UPLOAD_DIR / filename
    written_bytes = 0
    chunk_size = 10 * 1024 * 1024  # 10 MB streams

    try:
        with open(target_path, "wb") as buffer:
            while chunk := await file.read(chunk_size):
                written_bytes += len(chunk)
                if written_bytes > MAX_FILE_SIZE_BYTES:
                    buffer.close()
                    target_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="File size exceeds the maximum allowed limit of 20 GB.")
                buffer.write(chunk)
    except Exception as e:
        target_path.unlink(missing_ok=True)
        if isinstance(e, HTTPException):
            raise e
        logger.error("Upload failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to write file to disk: {str(e)}")

    return {
        "status": "success",
        "filename": filename,
        "size_human": _human_readable_size(written_bytes),
        "message": "File successfully stored in node path directory."
    }


@app.delete("/uploads")
def delete_upload(filename: str = Query(..., description="Target filename to remove")):
    """Deletes a file from the node storage directory."""
    safe_filename = Path(filename).name
    target_path = UPLOAD_DIR / safe_filename

    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="Requested file not found.")

    try:
        target_path.unlink()
        return {"status": "success", "message": f"Deleted file '{safe_filename}'."}
    except Exception as e:
        logger.error("Delete failure: %s", e)
        raise HTTPException(status_code=500, detail=f"Could not delete file: {str(e)}")


@app.get("/agents/route-preview")
def route_preview(message: str):
    """Debug endpoint — shows which agent would handle a message without calling LLM."""
    agent = detect_agent(message)
    return {
        "message": message,
        "detected_agent": agent,
        "model_url": model_for_agent(agent),
        "model_name": model_name_for_agent(agent),
    }
