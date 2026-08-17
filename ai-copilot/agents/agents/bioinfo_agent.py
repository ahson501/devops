"""
Bioinfo Agent
Handles genomics, structural biology, and sequence analysis queries.

Capabilities:
  1. Parse uploaded files from /data/research-uploads (FASTA, PDB, VCF, GFF)
  2. Search Qdrant: pubmed + biomodels collections
  3. Synthesise grounded answer via Qwen
"""

import os
import re
import logging
import httpx
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
EMBED_URL      = os.getenv("EMBED_URL",      "http://embedding-bge-svc.ai-models.svc.cluster.local:80/embed")
QDRANT_URL     = os.getenv("QDRANT_URL",     "http://qdrant-svc.ai-models.svc.cluster.local:6333")
UPLOAD_DIR     = Path(os.getenv("UPLOAD_DIR", "/data/research-uploads"))
HTTP_TIMEOUT   = float(os.getenv("HTTP_TIMEOUT", "15.0"))
LLM_TIMEOUT    = float(os.getenv("LLM_TIMEOUT",  "120.0"))
QDRANT_TOP_K   = int(os.getenv("QDRANT_TOP_K",   "5"))
MAX_TOKENS     = int(os.getenv("MAX_TOKENS",      "512"))
ABSTRACT_CAP   = int(os.getenv("ABSTRACT_CHAR_CAP", "200"))
CONTEXT_CAP    = int(os.getenv("CONTEXT_CHAR_CAP",  "1200"))

SYSTEM_PROMPT = (
    "You are an expert Bioinformatician and Structural Biologist. "
    "You have access to uploaded sequence/structure files and PubMed/BioModels literature. "
    "When analysing sequences: identify key features, mutations, conserved regions. "
    "When analysing structures: describe binding sites, secondary structure, functional domains. "
    "Cite evidence using [PMID:XXXXX] or [Model:XXXXX]. "
    "Structure: Biological Summary → File Analysis → Literature Evidence → Recommendations."
)

# ── File parsers ─────────────────────────────────────────────────────────────

def _parse_fasta(path: Path, max_seqs: int = 3, max_chars: int = 300) -> str:
    """Parse FASTA file — return first N sequences truncated."""
    try:
        content = path.read_text(errors="ignore")
        seqs = re.split(r'(?=>)', content.strip())
        result = []
        for seq in seqs[:max_seqs]:
            lines   = seq.strip().splitlines()
            header  = lines[0] if lines else ""
            sequence = "".join(lines[1:])[:max_chars]
            result.append(f"{header}\n{sequence}{'...' if len(''.join(lines[1:])) > max_chars else ''}")
        total = len(seqs)
        summary = f"[FASTA: {path.name} | {total} sequence(s)]\n" + "\n".join(result)
        return summary
    except Exception as e:
        return f"[FASTA parse error: {e}]"


def _parse_vcf(path: Path, max_variants: int = 10) -> str:
    """Parse VCF file — return header metadata + first N variants."""
    try:
        lines    = path.read_text(errors="ignore").splitlines()
        meta     = [l for l in lines if l.startswith("##")][:5]
        header   = next((l for l in lines if l.startswith("#CHROM")), "")
        variants = [l for l in lines if not l.startswith("#")][:max_variants]
        count    = sum(1 for l in lines if not l.startswith("#"))
        return (
            f"[VCF: {path.name} | {count} variants total]\n"
            f"Metadata: {'; '.join(m.lstrip('#') for m in meta)}\n"
            f"Columns: {header}\n"
            f"First {len(variants)} variants:\n" + "\n".join(variants)
        )
    except Exception as e:
        return f"[VCF parse error: {e}]"


def _parse_pdb(path: Path, max_chars: int = 600) -> str:
    """Parse PDB file — extract HEADER, TITLE, SEQRES, REMARK."""
    try:
        lines   = path.read_text(errors="ignore").splitlines()
        records = ["HEADER", "TITLE", "SEQRES", "REMARK", "ATOM"]
        kept    = []
        atom_count = 0
        for line in lines:
            rec = line[:6].strip()
            if rec in ["HEADER", "TITLE", "REMARK"]:
                kept.append(line[:80])
            elif rec == "SEQRES":
                kept.append(line[:80])
            elif rec == "ATOM":
                atom_count += 1
        summary = "\n".join(kept)[:max_chars]
        return f"[PDB: {path.name} | {atom_count} ATOM records]\n{summary}"
    except Exception as e:
        return f"[PDB parse error: {e}]"


def _parse_gff(path: Path, max_features: int = 10) -> str:
    """Parse GFF/GFF3 genome annotation file."""
    try:
        lines    = path.read_text(errors="ignore").splitlines()
        comments = [l for l in lines if l.startswith("#")][:3]
        features = [l for l in lines if not l.startswith("#")][:max_features]
        total    = sum(1 for l in lines if not l.startswith("#"))
        return (
            f"[GFF: {path.name} | {total} features total]\n"
            + "\n".join(comments) + "\n"
            + "\n".join(features)
        )
    except Exception as e:
        return f"[GFF parse error: {e}]"


def _parse_generic_text(path: Path, max_chars: int = 500) -> str:
    """Fallback parser for txt/csv/tsv files."""
    try:
        content = path.read_text(errors="ignore")[:max_chars]
        return f"[{path.suffix.upper()}: {path.name}]\n{content}"
    except Exception as e:
        return f"[Parse error: {e}]"


PARSERS = {
    ".fasta": _parse_fasta,
    ".fa":    _parse_fasta,
    ".fna":   _parse_fasta,
    ".ffn":   _parse_fasta,
    ".vcf":   _parse_vcf,
    ".pdb":   _parse_pdb,
    ".gff":   _parse_gff,
    ".gff3":  _parse_gff,
    ".txt":   _parse_generic_text,
    ".csv":   _parse_generic_text,
    ".tsv":   _parse_generic_text,
}


def _scan_uploads(message: str) -> str:
    """
    Scan /data/research-uploads for bioinformatics files.
    If message mentions a filename, prioritise that file.
    Otherwise return summary of all bioinfo files found.
    """
    if not UPLOAD_DIR.exists():
        return ""

    bio_extensions = set(PARSERS.keys())
    all_files = [
        f for f in UPLOAD_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in bio_extensions
    ]

    if not all_files:
        return ""

    # Check if user mentioned a specific filename
    msg_lower = message.lower()
    mentioned = [f for f in all_files if f.name.lower() in msg_lower]
    target_files = mentioned if mentioned else all_files[:3]  # max 3 files

    parsed = []
    for f in target_files:
        ext    = f.suffix.lower()
        parser = PARSERS.get(ext, _parse_generic_text)
        parsed.append(parser(f))

    header = f"=== UPLOADED FILES ({len(all_files)} bioinfo files in /data/research-uploads) ==="
    return header + "\n\n" + "\n\n".join(parsed)


# ── Qdrant helpers ────────────────────────────────────────────────────────────

async def _embed(text: str, client: httpx.AsyncClient) -> list[float] | None:
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
        logger.error("Embed failed: %s", e)
        return None


async def _qdrant_search(
    collection: str,
    vector: list[float],
    top_k: int,
    client: httpx.AsyncClient,
) -> list[dict[str, Any]]:
    try:
        resp = await client.post(
            f"{QDRANT_URL}/collections/{collection}/points/search",
            json={"vector": vector, "limit": top_k, "with_payload": True, "with_vector": False},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("result", [])
    except Exception as e:
        logger.error("Qdrant [%s] failed: %s", collection, e)
        return []


def _fmt_pubmed(hits: list[dict]) -> str:
    if not hits:
        return ""
    lines = ["=== PUBMED EVIDENCE ==="]
    for i, hit in enumerate(hits, 1):
        p   = hit.get("payload", {})
        ctx = str(p.get("context", p.get("abstract", "N/A")))[:ABSTRACT_CAP]
        lines.append(f"[{i}] PMID:{p.get('pmid','?')} \"{p.get('title','N/A')[:80]}\"\n    {ctx}")
    return "\n".join(lines)


def _fmt_biomodels(hits: list[dict]) -> str:
    if not hits:
        return ""
    lines = ["=== BIOMODELS ==="]
    for i, hit in enumerate(hits, 1):
        p   = hit.get("payload", {})
        ctx = str(p.get("context", p.get("description", "N/A")))[:150]
        lines.append(f"[{i}] {p.get('model_id','?')}: {ctx}")
    return "\n".join(lines)


async def _call_llm(
    messages: list[dict],
    model_url: str,
    model_name: str,
    client: httpx.AsyncClient,
) -> str:
    resp = await client.post(
        f"{model_url}/v1/chat/completions",
        json={
            "model":       model_name,
            "messages":    messages,
            "max_tokens":  MAX_TOKENS,
            "temperature": float(os.getenv("TEMPERATURE", "0.2")),
        },
        timeout=LLM_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ── Public entry point ────────────────────────────────────────────────────────

async def run(
    message: str,
    history: list[dict],
    model_url: str,
    model_name: str,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """
    Bioinfo RAG pipeline:
      1. Parse uploaded files (FASTA/PDB/VCF/GFF)
      2. Embed query → search Qdrant (pubmed + biomodels)
      3. Build grounded prompt → Qwen
    """
    should_close = client is None
    if client is None:
        client = httpx.AsyncClient()

    try:
        # 1. Scan uploaded files
        file_context = _scan_uploads(message)
        if file_context:
            logger.info("File context extracted from uploads (%d chars)", len(file_context))

        # 2. Embed + Qdrant
        vector         = await _embed(message, client)
        pubmed_hits:    list[dict] = []
        biomodels_hits: list[dict] = []

        if vector:
            pubmed_hits    = await _qdrant_search("pubmed",    vector, QDRANT_TOP_K, client)
            biomodels_hits = await _qdrant_search("biomodels", vector, 3,            client)
            logger.info("Qdrant hits — pubmed:%d biomodels:%d", len(pubmed_hits), len(biomodels_hits))

        # 3. Assemble context (file context + literature, strictly capped)
        lit_context = f"{_fmt_pubmed(pubmed_hits)}\n{_fmt_biomodels(biomodels_hits)}".strip()

        # Budget: file context gets priority, literature fills remaining space
        file_ctx_capped = file_context[:700] if file_context else ""
        lit_ctx_capped  = lit_context[:500]  if lit_context  else ""

        full_context = "\n\n".join(filter(None, [file_ctx_capped, lit_ctx_capped]))

        grounded_msg = (
            f"Bioinformatics Context:\n{full_context}\n\n"
            f"Question: {message[:300]}\n\n"
            f"Provide a precise biological analysis. Cite PMIDs and file data where relevant."
        )

        # 4. Build messages
        llm_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in history[-2:]:
            if turn.get("role") in ("user", "assistant"):
                llm_messages.append({"role": turn["role"], "content": turn["content"][:200]})
        llm_messages.append({"role": "user", "content": grounded_msg})

        # 5. LLM call
        try:
            answer = await _call_llm(llm_messages, model_url, model_name, client)
        except Exception as e:
            logger.error("LLM failed: %s", e)
            files_found = "yes" if file_context else "none"
            answer = (
                f"Files parsed: {files_found} | "
                f"PubMed hits: {len(pubmed_hits)} | "
                f"BioModels hits: {len(biomodels_hits)} | "
                f"LLM unavailable: {e}"
            )

        # 6. Sources
        sources = [
            {"type": "pubmed",   "pmid":     h.get("payload",{}).get("pmid"),     "title": h.get("payload",{}).get("title"), "score": round(h.get("score",0),3)}
            for h in pubmed_hits
        ] + [
            {"type": "biomodel", "model_id": h.get("payload",{}).get("model_id"), "name":  h.get("payload",{}).get("name"),  "score": round(h.get("score",0),3)}
            for h in biomodels_hits
        ]

        # Add parsed files to sources list
        if file_context:
            upload_files = [
                f for f in UPLOAD_DIR.iterdir()
                if f.is_file() and f.suffix.lower() in set(PARSERS.keys())
            ] if UPLOAD_DIR.exists() else []
            for f in upload_files[:3]:
                sources.append({"type": "uploaded_file", "filename": f.name, "path": str(f)})

        return {
            "answer":       answer,
            "sources":      sources,
            "agent":        "bioinfo",
            "model":        model_name,
            "context_hits": {
                "pubmed":        len(pubmed_hits),
                "biomodels":     len(biomodels_hits),
                "uploaded_files": 1 if file_context else 0,
            },
        }

    finally:
        if should_close:
            await client.aclose()
