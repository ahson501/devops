import requests
import time
import json
import re

# =========================
# MODEL ENDPOINTS
# =========================
MODELS = {
    "deepseek": "http://deepseek-coder-svc.ai-models.svc.cluster.local:8000/v1/chat/completions",
    "qwen": "http://qwen-7b-svc.ai-models.svc.cluster.local:8000/v1/chat/completions",
    "mistral": "http://mistral-7b-svc.ai-models.svc.cluster.local:8000/v1/chat/completions"
}

MODEL_NAMES = {
    "deepseek": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "qwen": "Qwen/Qwen2.5-3B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2"
}

# =========================
# SIMPLE MODEL HEALTH TRACKING
# =========================
MODEL_HEALTH = {
    "deepseek": {"failures": 0},
    "qwen": {"failures": 0},
    "mistral": {"failures": 0}
}


# =========================
# ROUTING LOGIC (INTELLIGENT)
# =========================
def get_model_priority(prompt: str):
    p = prompt.lower()

    # Incident / debugging → strongest reasoning model first
    if any(x in p for x in ["error", "debug", "crash", "failure", "down", "critical"]):
        return ["deepseek", "qwen", "mistral"]

    # Kubernetes / infra → structured reasoning
    if any(x in p for x in ["kubernetes", "pod", "deployment", "yaml", "hpa"]):
        return ["qwen", "deepseek", "mistral"]

    # Default balanced routing
    return ["qwen", "mistral", "deepseek"]


# =========================
# EXTRACT TEXT (ROBUST)
# =========================
def extract_text(response_json):
    try:
        return response_json["choices"][0]["message"]["content"]
    except Exception:
        return None


# =========================
# SMART JSON CLEANER (IMPORTANT FOR NOC AI)
# =========================
def try_extract_json(text):
    """
    Extract JSON even if model adds extra text
    """
    try:
        # direct parse
        return json.loads(text)
    except:
        pass

    try:
        # extract JSON block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except:
        pass

    return None


# =========================
# PROMPT HARDENING (CRITICAL)
# =========================
def build_system_prompt():
    return """
You are an Autonomous Network Operations Center (NOC) AI.

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
}

RULES:
- Use ONLY provided metrics and alerts
- If unsure, lower confidence
- Be precise and operational
"""


# =========================
# CALL MODEL (ROBUST + TIME TRACKING)
# =========================
def call_model(model_key: str, prompt: str):

    url = MODELS[model_key]

    payload = {
        "model": MODEL_NAMES[model_key],
        "messages": [
            {
                "role": "system",
                "content": build_system_prompt()
            },
            {
                "role": "user",
                "content": prompt
            }
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

        data = r.json()
        text = extract_text(data)

        if not text:
            MODEL_HEALTH[model_key]["failures"] += 1
            return None, None, latency

        return text, model_key, latency

    except Exception:
        MODEL_HEALTH[model_key]["failures"] += 1
        return None, None, None


# =========================
# MAIN AI ROUTER (STAGE 4)
# =========================
def run_ai(prompt: str):

    priority_list = get_model_priority(prompt)

    best_result = None

    for model in priority_list:

        text, used_model, latency = call_model(model, prompt)

        if text:

            structured = try_extract_json(text)

            # If JSON is valid → return immediately (BEST CASE)
            if structured:
                return {
                    "model_used": used_model,
                    "response": structured,
                    "latency": latency
                }

            # fallback: still return raw text but wrapped later in main.py
            best_result = {
                "model_used": used_model,
                "response": text,
                "latency": latency
            }

    if best_result:
        return best_result

    return {
        "error": "All AI models are unavailable"
    }
