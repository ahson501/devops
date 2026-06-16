import requests
import re
import time
from ai_router import route_scientific_query, run_ai, try_extract_json
from k8s_client import trigger_healing_action
from prometheus_client import (
    get_device_metrics,
    get_down_interfaces,
    get_error_interfaces
)

ALERT_URL = "http://alertmanager.monitoring.svc.cluster.local:9093/api/v2/alerts"

def get_alerts():
    try:
        return requests.get(ALERT_URL, timeout=5).json()
    except:
        return []

def extract_from_message(msg):
    pattern = r'\b[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)+\b'
    matches = re.findall(pattern, msg)
    if matches:
        return matches[0]
    return None

def anomaly_score(metrics):
    return min((metrics.get("interfaces_down", 0) * 20) + (metrics.get("in_errors", 0) * 10), 100)

def pre_check(metrics):
    if metrics.get("monitoring_status") == "no_snmp_data":
        return {"status": "WARNING", "root_cause": "No SNMP metrics available", "recommendation": "Check SNMP connectivity", "confidence": 0.98, "action": "check_monitoring"}
    if metrics.get("interfaces_down", 0) > 0:
        return {"status": "WARNING", "root_cause": "One or more important interfaces are DOWN", "recommendation": "Check links/SFPs", "confidence": 0.95, "action": "check_interfaces"}
    if metrics.get("in_errors", 0) > 10:
        return {"status": "WARNING", "root_cause": "High interface error rate detected", "recommendation": "Check cabling infrastructure", "confidence": 0.92, "action": "check_errors"}
    return {"status": "NORMAL", "root_cause": "No major issues detected", "recommendation": "No action needed", "confidence": 0.95, "action": "none"}

def safe_heal(device, action):
    if action == "none":
        return
    try:
        trigger_healing_action(device, action)
    except Exception as e:
        print(f"Healing failed: {e}")

def analyze_device(data):
    device = data.get("device")
    if not device:
        return {"error": "Device not provided"}
    metrics = get_device_metrics(device)
    analysis = pre_check(metrics)
    safe_heal(device, analysis.get("action"))
    return {"device": device, "metrics": metrics, "analysis": analysis, "score": anomaly_score(metrics), "model_used": "rule-engine"}

# =====================================================
# CORE INTERCEPTING CHAT HANDLER
# =====================================================
def chat_handler(data):
    msg = data.get("message", "")
    msg_lower = msg.lower()

    # 🔬 SCIENTIFIC DETECTOR INTERCEPTOR
    scientific_keywords = ["purity", "molecule", "compound", "pubmed", "pubchem", "synthesis", "reaction", "protein", "dna", "rna", "toxicity", "formula"]
    if any(keyword in msg_lower for keyword in scientific_keywords):
        target_model = "mistral"
        if any(b in msg_lower for b in ["dna", "rna", "protein", "disease", "clinical", "pubmed"]):
            target_model = "qwen"
        elif any(c in msg_lower for c in ["simulate", "model", "matlab", "biomodels", "equation"]):
            target_model = "deepseek"

        print(f"[ICCBS AI] Intercepted scientific query. Routing to {target_model} via RAG...")
        return route_scientific_query(msg, target_model)

    # 🖥️ INFRASTRUCTURE NOC WORKFLOWS
    device = data.get("device") or extract_from_message(msg_lower)
    if not device:
        return {"error": "Device not found or not a valid scientific query"}

    metrics = get_device_metrics(device)

    if "interface status" in msg_lower or "show interfaces" in msg_lower:
        return {"device": device, "interfaces_down": metrics.get("interfaces_down"), "total_interfaces": metrics.get("total_interfaces"), "down_interfaces": metrics.get("down_interfaces"), "source": "prometheus"}

    if "which interfaces" in msg_lower or "interfaces are down" in msg_lower or "down interfaces" in msg_lower:
        return {"device": device, "down_interfaces": metrics.get("down_interfaces"), "source": "prometheus"}

    if msg_lower.startswith("show errors") or msg_lower.startswith("check crc") or msg_lower.startswith("show crc"):
        return {"device": device, "error_interfaces": metrics.get("error_interfaces"), "source": "prometheus"}

    if "traffic" in msg_lower or "bandwidth" in msg_lower:
        query_focus = "Focus primarily on traffic utilization, throughput trends, and bandwidth."
        default_status = "NORMAL unless bandwidth is highly saturated"
    elif "hardware" in msg_lower or "health" in msg_lower or "power" in msg_lower or "fan" in msg_lower:
        query_focus = "Focus primarily on physical hardware health, fans, and power supplies."
        default_status = "NORMAL unless hardware sensors show a failure"
    else:
        query_focus = "Perform a general network triage analyzing administrative downs, operational downs, and errors."
        default_status = "WARNING/CRITICAL if any production interfaces are down or errors exist"

    prompt = f"""You are an elite NOC Triaging Engine analyzing telemetry from: {device}.
User's Intent: "{msg}"
Objective: {query_focus}
Telemetry:
- Total Interfaces: {metrics.get("total_interfaces")}
- Active DOWN: {metrics.get("interfaces_down")}
- Down List: {metrics.get("down_interfaces")}
- Traffic: In: {metrics.get("in_traffic_mbps", 0.0)} Mbps, Out: {metrics.get("out_traffic_mbps", 0.0)} Mbps
- Errors: In: {metrics.get("in_errors", 0)}, Out: {metrics.get("out_errors", 0)}
Rules: Establish status based on baseline rule: {default_status}. Return JSON only.
Required Format:
{{ "status": "NORMAL|WARNING|CRITICAL", "root_cause": "...", "recommendation": "...", "confidence": 0.0-1.0 }}"""
    
    return run_ai(prompt)
