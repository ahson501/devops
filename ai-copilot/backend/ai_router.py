import requests
import time
import json
import re
import os
import socket
import logging

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

def force_ipv4():
    """
    Forces socket address resolution to AF_INET (IPv4).
    This bypasses slow dual-stack IPv6 lookups within core Kubernetes DNS configurations.
    """
    orig = socket.getaddrinfo
    def wrapper(host, port, family=0, type=0, proto=0, flags=0):
        return orig(host, port, socket.AF_INET, type, proto, flags)
    socket.getaddrinfo = wrapper

force_ipv4()

# Establish a persistent session pool to keep connections warm and avoid TCP handshake overhead
http_session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=50)
http_session.mount("http://", adapter)
http_session.mount("https://", adapter)

# Dynamic imports from central cluster configurations
try:
    from config import (
        MISTRAL_URL,
        QWEN_URL,
        DEEPSEEK_URL
    )
except ImportError:
    # Safe fallbacks if run outside of custom environment settings
    logging.warning("Config module not found. Falling back to default cluster endpoints.")
    MISTRAL_URL = "http://mistral-svc.ai-models.svc.cluster.local:8000/v1"
    QWEN_URL = "http://qwen-svc.ai-models.svc.cluster.local:8000/v1"
    DEEPSEEK_URL = "http://deepseek-svc.ai-models.svc.cluster.local:8000/v1"

MODELS = {
    "deepseek": f"{DEEPSEEK_URL}/chat/completions",
    "qwen": f"{QWEN_URL}/chat/completions",
    "mistral": f"{MISTRAL_URL}/chat/completions",
}

MODEL_NAMES = {
    "deepseek": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "qwen": "Qwen/Qwen2.5-3B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2"
}

# Real-time node health monitoring registry
MODEL_HEALTH = {
    "deepseek": {"failures": 0},
    "qwen": {"failures": 0},
    "mistral": {"failures": 0}
}

MAX_ALLOWED_FAILURES = 3

# Scientific classification dictionaries mapping natural language prompts to database targets
CHEM_KEYWORDS = [
    "smiles", "inchi", "molecule", "molecular", "compound", "pubchem", 
    "cid", "quantum", "qm", "dft", "cp2k", "scf", "orbitals", "homo", 
    "lumo", "docking", "dock", "drug-target", "thermodynamic", "enthalpy", 
    "gibbs", "energy", "inhibitor", "binding affinity", "property prediction",
    "pbe", "tzvp", "molopt", "spin contamination"
]

SYSBIO_KEYWORDS = [
    "biomodels", "sbml", "kinetic", "kinetic modeling", "ode", "pde", 
    "differential equation", "pathway", "pathway simulation", "cellular", 
    "signaling", "metabolic", "metabolic flux", "flux analysis", 
    "population dynamics", "model_id", "apoptosis", "cell cycle", "feedback loop"
]

GENOMICS_KEYWORDS = [
    "clinical trial", "genomic", "gene", "sequence analysis", "mutation", 
    "variant", "protein folding", "egfr", "brca", "tp53", "pubmed", 
    "oncology", "literature", "biomedical"
]

def route_scientific_query(user_prompt: str, target_model: str) -> dict:
    """
    Renders RAG pipeline with embedding vectors and Qdrant integration.
    Includes active node failover handling for the downstream vLLM generator.
    """
    try:
        from prompts import SCIENTIFIC_RAG_TEMPLATE  
    except ImportError:
        SCIENTIFIC_RAG_TEMPLATE = "Context:\n{context}\n\nQuery:\n{query}"

    BGE_SERVICE_URL = "http://embedding-bge-svc.ai-models.svc.cluster.local:80/embed"
    QDRANT_BASE_URL = "http://qdrant-svc.ai-models.svc.cluster.local:6333"

    p = user_prompt.lower()
    
    # Context-driven keyword evaluation logic
    if any(x in p for x in CHEM_KEYWORDS):
        collection_name = "pubchem"
        system_role = (
            "You are an expert Computational Chemist and Quantum Mechanics Simulator "
            "backed by PubChem molecular databases, SMILES/InChI parsers, and DFT/QM solver context."
        )
    elif any(x in p for x in SYSBIO_KEYWORDS):
        collection_name = "biomodels"
        system_role = (
            "You are an expert Mathematical Systems Biology Simulator "
            "backed by BioModels SBML repositories, ODE/PDE differential solvers, and metabolic flux dynamics."
        )
    else:
        # Default fallback collection (incorporating PubMed / genomics keywords)
        collection_name = "pubmed"
        system_role = "You are an expert Medical & Biological Research Assistant backed by PubMed literature."

    retrieved_context = ""
    try:
        # Step 1: Query local BGE Embedding service
        bge_response = http_session.post(
            BGE_SERVICE_URL, 
            json={"inputs": user_prompt[:512]}, 
            timeout=15
        )
        if bge_response.status_code != 200:
            raise Exception(f"BGE vector generation failed with status {bge_response.status_code}")
        
        query_vector = bge_response.json()[0]

        # Step 2: Query Qdrant Vector database using retrieved float coordinate array
        qdrant_search_url = f"{QDRANT_BASE_URL}/collections/{collection_name}/points/search"
        qdrant_payload = {
            "vector": query_vector,
            "limit": 3,
            "with_payload": True
        }
        
        qdrant_response = http_session.post(
            qdrant_search_url, 
            json=qdrant_payload, 
            timeout=10
        )
        if qdrant_response.status_code == 200:
            hits = qdrant_response.json().get("result", [])
            retrieved_context = "\n\n".join([
                hit["payload"]["context"] for hit in hits 
                if hit.get("payload") and "context" in hit["payload"]
            ])

    except Exception as e:
        logging.warning(f"Failed vector pipeline extraction sequence: {e}. Triggering guardrail short-circuit.")
        return get_guardrail_response()

    # Short-circuit execution if context is empty or corrupted to protect generation bounds
    if not retrieved_context or len(retrieved_context.strip()) < 10:
        logging.warning("[ICCBS GUARDRAIL] Retrieved context is empty or failing length validation bounds. Intercepting.")
        return get_guardrail_response()

    # Step 3: Run target model with automatic, resilient failover mechanisms
    formatted_user_content = SCIENTIFIC_RAG_TEMPLATE.format(context=retrieved_context, query=user_prompt)
    
    # Establish priority list starting with user's selected node
    priority_list = [target_model] + [m for m in MODELS if m != target_model]
    
    for model_key in priority_list:
        # Skip node if currently offline/flagged unhealthy
        if MODEL_HEALTH[model_key]["failures"] >= MAX_ALLOWED_FAILURES:
            logging.warning(f"Skipping unhealthy node {model_key} during scientific query invocation.")
            continue

        endpoint = MODELS[model_key]
        real_name = MODEL_NAMES[model_key]

        payload = {
            "model": real_name,
            "messages": [
                {"role": "system", "content": system_role},
                {"role": "user", "content": formatted_user_content}
            ],
            "temperature": 0.0  
        }

        try:
            start_time = time.time()
            response = http_session.post(endpoint, json=payload, timeout=60)
            latency = time.time() - start_time

            if response.status_code != 200:
                raise Exception(f"API endpoint returned non-200 status code: {response.status_code}")

            raw_text = response.json()["choices"][0]["message"]["content"]
            
            # Reset health tracker on a successful request execution
            MODEL_HEALTH[model_key]["failures"] = 0

            return {
                "model_used": real_name,
                "response": {"status": "SUCCESS", "scientific_output": raw_text},
                "latency": latency,
                "context_retrieved": True
            }

        except Exception as e:
            MODEL_HEALTH[model_key]["failures"] += 1
            logging.error(f"vLLM node invocation failed on {model_key}: {e}. Trying alternate fallback route...")

    return {"error": "All AI core generators are currently offline or unavailable."}


def get_guardrail_response() -> dict:
    """Standardized guardrail output representation."""
    return {
        "model_used": "system-guardrail",
        "response": {
            "status": "SUCCESS", 
            "scientific_output": "The local institutional database does not contain this information."
        },
        "latency": 0.0,
        "context_retrieved": False
    }


def get_model_priority(prompt: str) -> list:
    """Categorizes prompt strings to route to specialized local model engines."""
    p = prompt.lower()
    if any(x in p for x in ["error", "debug", "crash", "failure", "down", "critical"]):
        return ["deepseek", "qwen", "mistral"]
    if any(x in p for x in ["kubernetes", "pod", "deployment", "yaml", "hpa"]):
        return ["qwen", "deepseek", "mistral"]
    return ["qwen", "mistral", "deepseek"]


def extract_text(response_json: dict) -> str:
    try:
        return response_json["choices"][0]["message"]["content"]
    except Exception:
        return None


def try_extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass
    return None


def build_system_prompt() -> str:
    return """You are an Autonomous Network Operations Center (NOC) AI.
STRICT OUTPUT RULES:
- You MUST respond ONLY in valid JSON
- No explanations, no markdown, no text outside JSON
REQUIRED FORMAT:
{
  "status": "CRITICAL | WARNING | NORMAL",
  "root_cause": "...",
  "recommendation": "...",
  "confidence": 0.0-1.0,
  "action": "none | check | restart_interface | restart_pod | scale_up"
}"""


def call_model(model_key: str, prompt: str) -> tuple:
    """Invocates a single model using persistent pool connection."""
    url = MODELS[model_key]
    payload = {
        "model": MODEL_NAMES[model_key],
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1
    }
    try:
        start = time.time()
        r = http_session.post(url, json=payload, timeout=40)
        latency = time.time() - start
        
        if r.status_code != 200:
            MODEL_HEALTH[model_key]["failures"] += 1
            return None, None, latency
            
        text = extract_text(r.json())
        if not text:
            MODEL_HEALTH[model_key]["failures"] += 1
            return None, None, latency
            
        # Reset health state tracker on success
        MODEL_HEALTH[model_key]["failures"] = 0
        return text, model_key, latency

    except Exception:
        MODEL_HEALTH[model_key]["failures"] += 1
        return None, None, None


def run_ai(prompt: str) -> dict:
    """Executes target priority models sequentially to handle complex network alerts."""
    priority_list = get_model_priority(prompt)
    best_result = None

    for model in priority_list:
        # Check system node health registers
        if MODEL_HEALTH[model]["failures"] >= MAX_ALLOWED_FAILURES:
            logging.warning(f"Skipping degraded node {model} in NOC priority loop.")
            continue

        text, used_model, latency = call_model(model, prompt)
        if text:
            structured = try_extract_json(text)
            if structured:
                return {
                    "model_used": MODEL_NAMES[used_model],
                    "response": structured,
                    "latency": latency
                }
            best_result = {
                "model_used": MODEL_NAMES[used_model],
                "response": text,
                "latency": latency
            }
            
    if best_result:
        return best_result
        
    return {"error": "All backend AI pipeline models are currently degraded or offline."}
