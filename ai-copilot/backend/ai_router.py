import requests
import time
import json
import re
import os
import socket

def force_ipv4():
    orig = socket.getaddrinfo
    def wrapper(host, port, family=0, type=0, proto=0, flags=0):
        return orig(host, port, socket.AF_INET, type, proto, flags)
    socket.getaddrinfo = wrapper

force_ipv4()
# =====================================================
# CORE ENDPOINTS & REAL NAMES
# =====================================================

from config import (
    MISTRAL_URL,
    QWEN_URL,
    DEEPSEEK_URL
)

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

MODEL_HEALTH = {
    "deepseek": {"failures": 0},
    "qwen": {"failures": 0},
    "mistral": {"failures": 0}
}

# =====================================================
# SCIENTIFIC PIPELINE ROUTER
# =====================================================
# =====================================================
# SCIENTIFIC PIPELINE ROUTER (FIXED FOR QDRANT & BGE)
# =====================================================
def route_scientific_query(user_prompt, target_model):
    from prompts import SCIENTIFIC_RAG_TEMPLATE  

    BGE_SERVICE_URL = "http://embedding-bge-svc.ai-models.svc.cluster.local:80/embed"
    QDRANT_BASE_URL = "http://qdrant-svc.ai-models.svc.cluster.local:6333"

    # 1. Inspect the incoming prompt keywords to choose the database collection
    p = user_prompt.lower()
    if any(x in p for x in ["molecule", "compound", "pubchem", "smiles", "drug", "docking", "inhibitor"]):
        collection_name = "pubchem"
        system_role = "You are an expert Computational Chemist backed by PubChem molecular data."
    elif any(x in p for x in ["biomodels", "sbml", "pathway", "simulation", "cellular", "signaling"]):
        collection_name = "biomodels"
        system_role = "You are an expert Systems Biology Simulator backed by BioModels repositories."
    else:
        # Default fallback collection for generalized biomedical queries
        collection_name = "pubmed"
        system_role = "You are an expert Medical & Biological Research Assistant backed by PubMed literature."

    # 2. Make sure the requested model engine is valid before wasting time on vectorizing
    if target_model not in MODELS:
        return {"error": f"Invalid model selection: {target_model}"}

    retrieved_context = ""
    try:
        # Step 1: Transform plain text prompt into a vector float array via BGE service
        bge_response = requests.post(BGE_SERVICE_URL, json={"inputs": user_prompt[:512]}, timeout=15)
        if bge_response.status_code != 200:
            raise Exception(f"BGE vector generation failed with status {bge_response.status_code}")
        
        query_vector = bge_response.json()[0]

        # Step 2: Query Qdrant with the generated vector array
        qdrant_search_url = f"{QDRANT_BASE_URL}/collections/{collection_name}/points/search"
        qdrant_payload = {
            "vector": query_vector,
            "limit": 3,
            "with_payload": True
        }
        
        qdrant_response = requests.post(qdrant_search_url, json=qdrant_payload, timeout=10)
        if qdrant_response.status_code == 200:
            hits = qdrant_response.json().get("result", [])
            # Combine the abstract/description contexts stored by your CronJob pipeline
            retrieved_context = "\n\n".join([
                hit["payload"]["context"] for hit in hits 
                if hit.get("payload") and "context" in hit["payload"]
            ])

    except Exception as e:
        print(f"[Warning] Failed vector pipeline extraction sequence: {e}.")
        return {
            "model_used": "system-guardrail",
            "response": {"status": "SUCCESS", "scientific_output": "The local institutional database does not contain this information."},
            "latency": 0.0,
            "context_retrieved": False
        }

    # =====================================================
    # CRITICAL HARD CODE-LEVEL GUARDRAIL INTERCEPT
    # =====================================================
    if not retrieved_context or len(retrieved_context.strip()) < 10:
        print(f"[ICCBS GUARDRAIL] Vector database context is empty for query. Short-circuiting execution.")
        return {
            "model_used": "system-guardrail",
            "response": {"status": "SUCCESS", "scientific_output": "The local institutional database does not contain this information."},
            "latency": 0.0,
            "context_retrieved": False
        }

    # Step 3: Pass user prompt + retrieved context to the final vLLM target instance
    formatted_user_content = SCIENTIFIC_RAG_TEMPLATE.format(context=retrieved_context, query=user_prompt)
    endpoint = MODELS[target_model]
    real_name = MODEL_NAMES[target_model]

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
        response = requests.post(endpoint, json=payload, timeout=60)
        latency = time.time() - start_time
        raw_text = response.json()["choices"][0]["message"]["content"]

        return {
            "model_used": real_name,
            "response": {"status": "SUCCESS", "scientific_output": raw_text},
            "latency": latency,
            "context_retrieved": True
        }
    except Exception as e:
        return {"error": f"vLLM invocation failed: {str(e)}"}
# =====================================================
# ROUTING LOGIC (INTELLIGENT NOC)
# =====================================================
def get_model_priority(prompt: str):
    p = prompt.lower()
    if any(x in p for x in ["error", "debug", "crash", "failure", "down", "critical"]):
        return ["deepseek", "qwen", "mistral"]
    if any(x in p for x in ["kubernetes", "pod", "deployment", "yaml", "hpa"]):
        return ["qwen", "deepseek", "mistral"]
    return ["qwen", "mistral", "deepseek"]

def extract_text(response_json):
    try:
        return response_json["choices"][0]["message"]["content"]
    except Exception:
        return None

def try_extract_json(text):
    try:
        return json.loads(text)
    except:
        pass
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except:
        pass
    return None

def build_system_prompt():
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

def call_model(model_key: str, prompt: str):
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
        r = requests.post(url, json=payload, timeout=40)
        latency = time.time() - start
        if r.status_code != 200:
            MODEL_HEALTH[model_key]["failures"] += 1
            return None, None, latency
        text = extract_text(r.json())
        if not text:
            MODEL_HEALTH[model_key]["failures"] += 1
            return None, None, latency
        return text, model_key, latency
    except Exception:
        MODEL_HEALTH[model_key]["failures"] += 1
        return None, None, None

def run_ai(prompt: str):
    priority_list = get_model_priority(prompt)
    best_result = None
    for model in priority_list:
        text, used_model, latency = call_model(model, prompt)
        if text:
            structured = try_extract_json(text)
            if structured:
                return {
                    "model_used": used_model,
                    "response": structured,
                    "latency": latency
                }
            best_result = {
                "model_used": used_model,
                "response": text,
                "latency": latency
            }
    if best_result:
        return best_result
    return {"error": "All AI models are unavailable"}
