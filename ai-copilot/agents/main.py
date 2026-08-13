"""
AI Agents Microservice  —  /agents/*
Separate pod from the main AI-NOC gateway.
Exposes a single /agents/chat endpoint that orchestrates all specialist agents.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from orchestrator import detect_agent, model_for_agent, model_name_for_agent
from agents import research_agent
# Future imports (uncomment as you build each):
# from agents import bioinfo_agent, chemistry_agent, coding_agent, statistics_agent

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
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
            "bioinfo":    {"model": model_name_for_agent("bioinfo"),     "status": "coming_soon"},
            "chemistry":  {"model": model_name_for_agent("chemistry"),   "status": "coming_soon"},
            "coding":     {"model": model_name_for_agent("coding"),      "status": "coming_soon"},
            "statistics": {"model": model_name_for_agent("statistics"),  "status": "coming_soon"},
            "general":    {"model": model_name_for_agent("general"),     "status": "active"},
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

    # ── Dispatch ─────────────────────────────────────────────────────────────
    try:
        if agent == "research":
            result = await research_agent.run(req.message, history, model_url, model_name)

        # Add remaining agents here as you build them:
        # elif agent == "bioinfo":
        #     result = await bioinfo_agent.run(req.message, history, model_url, model_name)
        # elif agent == "chemistry":
        #     result = await chemistry_agent.run(req.message, history, model_url, model_name)
        # elif agent == "coding":
        #     result = await coding_agent.run(req.message, history, model_url, model_name)
        # elif agent == "statistics":
        #     result = await statistics_agent.run(req.message, history, model_url, model_name)

        else:
            # General fallback — direct LLM call, no RAG
            import httpx
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
