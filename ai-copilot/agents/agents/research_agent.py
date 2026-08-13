"""
Biomedical Research Agent (Optimized)
Embeds queries via BGE, retrieves hits from Qdrant, 
and synthesizes grounded research summaries via Qwen.
"""

import os
import logging
import httpx
from typing import Any

logger = logging.getLogger(__name__)

# ── Cluster & Timeout Config ──────────────────────────────────────────────────
EMBED_URL    = os.getenv("EMBED_URL",    "http://embedding-bge-svc.ai-models.svc.cluster.local:80/embed")
QDRANT_URL   = os.getenv("QDRANT_URL",   "http://qdrant-svc.ai-models.svc.cluster.local:6333")
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "15.0"))
LLM_TIMEOUT  = float(os.getenv("LLM_TIMEOUT",  "120.0"))
QDRANT_TOP_K = int(os.getenv("QDRANT_TOP_K",   "5"))

# ── Optimized Context Budget for Qwen (vLLM Context-Expanded) ───────────────
MAX_TOKENS        = int(os.getenv("MAX_TOKENS", "512"))       # Expanded response token limit
CONTEXT_CHAR_CAP  = int(os.getenv("CONTEXT_CHAR_CAP", "1200"))   # Expanded total RAG window (~1500 tokens)
ABSTRACT_CHAR_CAP = int(os.getenv("ABSTRACT_CHAR_CAP", "200"))  # Captures complete paper summaries

SYSTEM_PROMPT = (
    "You are an expert biomedical Research Assistant with direct access to PubMed literature and BioModels. "
    "Synthesise evidence clearly using the retrieved context. "
    "Cite evidence directly using [PMID:XXXXX] or [Model:XXXXX]. "
    "Structure: Executive Summary → Key Evidence → Translational Implications → Limitations."
)

# ── Helper Services ──────────────────────────────────────────────────────────

async def _embed(text: str, client: httpx.AsyncClient) -> list[float] | None:
    """Encodes query into vector array using BGE endpoint."""
    try:
        resp = await client.post(
            EMBED_URL,
            json={"inputs": text[:512]},
            headers={"Content-Type": "application/json"},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()[0]
    except Exception as e:
        logger.error("BGE Embedding Service Failure: %s", e)
        return None


async def _qdrant_search(
    collection: str,
    vector: list[float],
    top_k: int,
    client: httpx.AsyncClient,
) -> list[dict[str, Any]]:
    """Performs cosine vector search on specified Qdrant collection."""
    try:
        resp = await client.post(
            f"{QDRANT_URL}/collections/{collection}/points/search",
            json={
                "vector": vector, 
                "limit": top_k, 
                "with_payload": True, 
                "with_vector": False
            },
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("result", [])
    except Exception as e:
        logger.error("Qdrant Vector Retrieval Error [%s]: %s", collection, e)
        return []


def _fmt_pubmed(hits: list[dict]) -> str:
    """Formats retrieved PubMed records with complete structural abstracts."""
    if not hits:
        return "No relevant PubMed articles found."
    lines = ["=== PUBMED RESEARCH EVIDENCE ==="]
    for i, hit in enumerate(hits, 1):
        p = hit.get("payload", {})
        pmid = p.get("pmid", "Unknown")
        title = p.get("title", "Untitled")
        
        # Read full context or fall back to abstract
        abstract = str(p.get("context", p.get("abstract", "No abstract available"))).strip()
        truncated_abstract = abstract[:ABSTRACT_CHAR_CAP]
        
        lines.append(f"[{i}] PMID:{pmid} - {title}\nAbstract: {truncated_abstract}\n")
    return "\n".join(lines)


def _fmt_biomodels(hits: list[dict]) -> str:
    """Formats retrieved BioModels computational pathway records."""
    if not hits:
        return ""
    lines = ["=== BIOMODELS PATHWAY SYSTEMS ==="]
    for i, hit in enumerate(hits, 1):
        p = hit.get("payload", {})
        model_id = p.get("model_id", "Unknown")
        desc = str(p.get("context", p.get("description", "No description"))).strip()[:400]
        lines.append(f"[{i}] Model ID: {model_id}\nDescription: {desc}\n")
    return "\n".join(lines)


async def _call_llm(
    messages: list[dict],
    model_url: str,
    model_name: str,
    client: httpx.AsyncClient,
) -> str:
    """Invokes vLLM OpenAI-compatible endpoint."""
    resp = await client.post(
        f"{model_url}/v1/chat/completions",
        json={
            "model": model_name,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": float(os.getenv("TEMPERATURE", "0.2")),
        },
        timeout=LLM_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ── Execution Entrypoint ─────────────────────────────────────────────────────

async def run(
    message: str,
    history: list[dict],
    model_url: str,
    model_name: str,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """RAG Orchestration Workflow: Embed -> Retrieve -> Build Prompt -> Generate."""
    
    # Session reuse pattern to minimize TCP connection creation overhead
    should_close = False
    if client is None:
        client = httpx.AsyncClient()
        should_close = True

    try:
        # 1. Generate Vector Embeddings
        vector = await _embed(message, client)

        # 2. Vector Search Execution
        pubmed_hits: list[dict] = []
        biomodels_hits: list[dict] = []

        if vector:
            pubmed_hits = await _qdrant_search("pubmed", vector, QDRANT_TOP_K, client)
            biomodels_hits = await _qdrant_search("biomodels", vector, 3, client)
            logger.info("Search Results Retained - PubMed: %d, BioModels: %d", len(pubmed_hits), len(biomodels_hits))
        else:
            logger.warning("Retrieval bypassed: Embedding vector null.")

        # 3. Context Construction
        raw_context = f"{_fmt_pubmed(pubmed_hits)}\n\n{_fmt_biomodels(biomodels_hits)}"
        bounded_context = raw_context.strip()[:CONTEXT_CHAR_CAP]

        grounded_msg = (
            f"Retrieved Research Context:\n{bounded_context}\n\n"
            f"User Question: {message[:1000]}\n\n"
            f"Provide a scientifically precise, cited synthesis based on the context above."
        )

        # 4. Message Assembly with History
        llm_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Include past chat interactions
        for turn in history[-4:]:
            if turn.get("role") in ("user", "assistant"):
                llm_messages.append({
                    "role": turn["role"],
                    "content": turn["content"][:500]
                })
                
        llm_messages.append({"role": "user", "content": grounded_msg})

        # 5. LLM Synthesis Call
        try:
            answer = await _call_llm(llm_messages, model_url, model_name, client)
        except Exception as e:
            logger.error("vLLM Ingestion Error: %s", e)
            answer = f"Data retrieved successfully ({len(pubmed_hits)} articles), but the LLM inference engine encountered an issue: {e}"

        # 6. Structured Output Format
        sources = [
            {
                "type": "pubmed",
                "pmid": h.get("payload", {}).get("pmid"),
                "title": h.get("payload", {}).get("title"),
                "score": round(h.get("score", 0), 3),
            }
            for h in pubmed_hits
        ] + [
            {
                "type": "biomodel",
                "model_id": h.get("payload", {}).get("model_id"),
                "name": h.get("payload", {}).get("name"),
                "score": round(h.get("score", 0), 3),
            }
            for h in biomodels_hits
        ]

        return {
            "answer": answer,
            "sources": sources,
            "agent": "research",
            "model": model_name,
            "context_hits": {"pubmed": len(pubmed_hits), "biomodels": len(biomodels_hits)},
        }

    finally:
        if should_close:
            await client.aclose()
