"""
Agent Orchestrator
Detects intent from user message and routes to the correct specialist agent.
Reads routing keywords from environment (mirrors ai-gateway-configmap.yaml logic).
"""

import os
import re
import logging
from typing import Literal

logger = logging.getLogger(__name__)

# ── Agent type literal ──────────────────────────────────────────────────────
AgentName = Literal["research", "bioinfo", "chemistry", "coding", "statistics", "general"]

# ── Keyword maps (extend these freely) ─────────────────────────────────────
AGENT_KEYWORDS: dict[AgentName, list[str]] = {
    "coding": [
        "python", "code", "script", "error", "debug", "debugging",
        "kubernetes", "docker", "bash", "pipeline", "function", "class",
        "import", "syntax", "exception", "traceback", "kubernetes", "kubectl",
        "yaml", "json", "api", "endpoint", "fastapi", "flask", "django",
        "sql", "query", "dataframe", "pandas", "numpy", "install", "pip",
    ],
    "bioinfo": [
        "genome", "dna", "rna", "protein", "bioinformatics", "gromacs",
        "molecular", "ligand", "receptor", "docking", "fasta", "fastq",
        "vcf", "gff", "pdb", "blast", "sequence", "alignment", "mutation",
        "snp", "variant", "gene", "chromosome", "transcriptome", "proteome",
        "alphafold", "structure", "crispr", "primer", "nucleotide", "codon",
    ],
    "chemistry": [
        "drug", "cheminformatics", "molecule", "compound", "smiles", "inchi",
        "mol", "sdf", "pubchem", "rdkit", "pharmacology", "bioavailability",
        "toxicity", "admet", "ic50", "ki", "binding", "pharmacokinetics",
        "synthesis", "reaction", "reagent", "solubility", "logp", "mw",
        "erlotinib", "gefitinib", "osimertinib", "kinase", "inhibitor",
    ],
    "statistics": [
        "statistics", "statistical", "regression", "correlation", "p-value",
        "pvalue", "anova", "t-test", "chi-square", "hypothesis", "plot",
        "graph", "chart", "visualize", "visualization", "histogram", "boxplot",
        "scatter", "heatmap", "pca", "clustering", "machine learning", "ml",
        "random forest", "svm", "neural", "accuracy", "precision", "recall",
        "r-squared", "distribution", "sample size", "confidence interval",
    ],
    "research": [
        "paper", "article", "study", "literature", "pubmed", "journal",
        "review", "meta-analysis", "clinical trial", "evidence", "cite",
        "reference", "publication", "research", "findings", "results",
        "oncology", "cancer", "therapy", "treatment", "disease", "biomarker",
        "pathway", "mechanism", "signaling", "egfr", "apoptosis", "cell cycle",
    ],
}

# Priority order — most specific first
AGENT_PRIORITY: list[AgentName] = [
    "coding",
    "bioinfo",
    "chemistry",
    "statistics",
    "research",
    "general",
]


def _score(text: str, keywords: list[str]) -> int:
    """Count how many keywords appear in text (whole-word, case-insensitive)."""
    text_lower = text.lower()
    return sum(
        1 for kw in keywords
        if re.search(rf"\b{re.escape(kw)}\b", text_lower)
    )


def detect_agent(message: str, history: list[dict] | None = None) -> AgentName:
    """
    Detect which agent should handle this message.

    Uses the current message plus optional recent history for context.
    Returns the agent name with the highest keyword score; falls back to
    'general' (→ Mistral) when no agent scores above zero.

    Args:
        message:  The user's current message.
        history:  Optional list of prior turns: [{"role": ..., "content": ...}]

    Returns:
        AgentName string.
    """
    # Combine current message with last 2 turns of history for context
    context = message
    if history:
        for turn in history[-2:]:
            context += " " + turn.get("content", "")

    scores: dict[AgentName, int] = {
        agent: _score(context, kws)
        for agent, kws in AGENT_KEYWORDS.items()
    }

    logger.debug("Intent scores: %s", scores)

    # Pick highest scoring agent; on tie, respect AGENT_PRIORITY order
    best_agent: AgentName = "general"
    best_score = 0

    for agent in AGENT_PRIORITY:
        if agent == "general":
            continue
        s = scores.get(agent, 0)
        if s > best_score:
            best_score = s
            best_agent = agent

    if best_score == 0:
        best_agent = "general"

    logger.info("Routing '%s...' → %s (score=%d)", message[:60], best_agent, best_score)
    return best_agent


def model_for_agent(agent: AgentName) -> str:
    """
    Return the vLLM base URL for the given agent.
    Reads from environment so ConfigMap values override defaults.
    """
    mapping: dict[AgentName, str] = {
        "coding":     os.getenv("DEEPSEEK_URL", "http://deepseek-coder-svc.ai-models.svc.cluster.local:8000"),
        "bioinfo":    os.getenv("QWEN_URL",     "http://qwen-7b-svc.ai-models.svc.cluster.local:8000"),
        "chemistry":  os.getenv("QWEN_URL",     "http://qwen-7b-svc.ai-models.svc.cluster.local:8000"),
        "statistics": os.getenv("QWEN_URL",     "http://qwen-7b-svc.ai-models.svc.cluster.local:8000"),
        "research":   os.getenv("QWEN_URL",     "http://qwen-7b-svc.ai-models.svc.cluster.local:8000"),
        "general":    os.getenv("MISTRAL_URL",  "http://mistral-7b-svc.ai-models.svc.cluster.local:8000"),
    }
    return mapping[agent]


def model_name_for_agent(agent: AgentName) -> str:
    """Return the model identifier string expected by vLLM's /v1/chat/completions."""
    mapping: dict[AgentName, str] = {
        "coding":     os.getenv("DEEPSEEK_MODEL", "deepseek-ai/deepseek-coder-6.7b-instruct"),
        "bioinfo":    os.getenv("QWEN_MODEL",     "Qwen/Qwen2.5-7B-Instruct"),
        "chemistry":  os.getenv("QWEN_MODEL",     "Qwen/Qwen2.5-7B-Instruct"),
        "statistics": os.getenv("QWEN_MODEL",     "Qwen/Qwen2.5-7B-Instruct"),
        "research":   os.getenv("QWEN_MODEL",     "Qwen/Qwen2.5-7B-Instruct"),
        "general":    os.getenv("MISTRAL_MODEL",  "mistralai/Mistral-7B-Instruct-v0.3"),
    }
    return mapping[agent]
