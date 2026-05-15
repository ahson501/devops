import requests
import re

from k8s_client import trigger_healing_action
from ai_router import run_ai

from prometheus_client import (
    get_device_metrics,
    get_down_interfaces,
    get_error_interfaces
)

ALERT_URL = "http://alertmanager.monitoring.svc.cluster.local:9093/api/v2/alerts"


# =====================================================
# ALERTS
# =====================================================
def get_alerts():
    try:
        return requests.get(ALERT_URL, timeout=5).json()
    except:
        return []


# =====================================================
# DEVICE EXTRACTION
# =====================================================
def extract_from_message(msg):

    # Match hostname-like patterns
    pattern = r'\b[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)+\b'

    matches = re.findall(pattern, msg)

    if matches:
        return matches[0]

    return None

# =====================================================
# ANOMALY SCORE
# =====================================================
def anomaly_score(metrics):

    return min(
        (metrics.get("interfaces_down", 0) * 20) +
        (metrics.get("in_errors", 0) * 10),
        100
    )


# =====================================================
# RULE ENGINE
# =====================================================
def pre_check(metrics):

    if metrics.get("monitoring_status") == "no_snmp_data":
        return {
            "status": "WARNING",
            "root_cause": "No SNMP metrics available",
            "recommendation": "Check SNMP exporter/device connectivity",
            "confidence": 0.98,
            "action": "check_monitoring"
        }

    if metrics.get("interfaces_down", 0) > 0:
        return {
            "status": "WARNING",
            "root_cause": "One or more important interfaces are DOWN",
            "recommendation": "Check uplinks/trunks/SFPs",
            "confidence": 0.95,
            "action": "check_interfaces"
        }

    if metrics.get("in_errors", 0) > 10:
        return {
            "status": "WARNING",
            "root_cause": "High interface error rate detected",
            "recommendation": "Check cabling/SFP/duplex mismatch",
            "confidence": 0.92,
            "action": "check_errors"
        }

    return {
        "status": "NORMAL",
        "root_cause": "No major issues detected",
        "recommendation": "No action needed",
        "confidence": 0.95,
        "action": "none"
    }


# =====================================================
# SAFE HEALING
# =====================================================
def safe_heal(device, action):

    if action == "none":
        return

    try:
        trigger_healing_action(device, action)
    except Exception as e:
        print(f"Healing failed: {e}")


# =====================================================
# ANALYZE DEVICE
# =====================================================
def analyze_device(data):

    device = data.get("device")

    if not device:
        return {"error": "Device not provided"}

    metrics = get_device_metrics(device)

    analysis = pre_check(metrics)

    safe_heal(device, analysis.get("action"))

    return {
        "device": device,
        "metrics": metrics,
        "analysis": analysis,
        "score": anomaly_score(metrics),
        "model_used": "rule-engine"
    }


# =====================================================
# CHAT HANDLER
# =====================================================
def chat_handler(data):

    msg = data.get("message", "").lower()

    device = data.get("device") or extract_from_message(msg)

    if not device:
        return {"error": "Device not found"}

    metrics = get_device_metrics(device)

    # =====================================================
    # INTERFACE STATUS
    # =====================================================
    if (
        "interface status" in msg or
        "show interfaces" in msg
    ):

        return {
            "device": device,
            "interfaces_down": metrics.get("interfaces_down"),
            "total_interfaces": metrics.get("total_interfaces"),
            "down_interfaces": metrics.get("down_interfaces"),
            "source": "prometheus"
        }

    # =====================================================
    # WHICH INTERFACES DOWN
    # =====================================================
    if (
        "which interfaces" in msg or
        "interfaces are down" in msg or
        "down interfaces" in msg
    ):

        return {
            "device": device,
            "down_interfaces": metrics.get("down_interfaces"),
            "source": "prometheus"
        }

    # =====================================================
    # ERROR INTERFACES
    # =====================================================

    if (
    msg.startswith("show errors") or
    msg.startswith("check crc") or
    msg.startswith("show crc")
    ):
        return {
            "device": device,
            "error_interfaces": metrics.get("error_interfaces"),
            "source": "prometheus"
        }

    # =====================================================
    # DYNAMIC AI TELEMETRY CONTEXT
    # =====================================================
    # Detect query type to focus the LLM's brain on the right objective
    if "traffic" in msg or "bandwidth" in msg:
        query_focus = "Focus primarily on traffic utilization, throughput trends, and bandwidth."
        default_status = "NORMAL unless bandwidth is highly saturated"
    elif "hardware" in msg or "health" in msg or "power" in msg or "fan" in msg:
        query_focus = "Focus primarily on physical hardware health, fans, and power supplies."
        default_status = "NORMAL unless hardware sensors show a failure (status code != 1)"
    else:
        query_focus = "Perform a general network triage analyzing administrative downs, operational downs, and errors."
        default_status = "WARNING/CRITICAL if any production interfaces are down or errors exist"

    # Build prompt dynamically without hardcoding a specific switch model
    prompt = f"""
You are an elite NOC (Network Operations Center) Triaging Engine. 
You are analyzing live SNMP telemetry from network device: {device}.

User's Intent/Query: "{data.get('message')}"
Your Focus Objective: {query_focus}

Raw Telemetry Received:
-----------------------------------------
- Total Interfaces Found: {metrics.get("total_interfaces")}
- Active DOWN/Shutdown Interfaces: {metrics.get("interfaces_down")}
- Interfaces list currently DOWN: {metrics.get("down_interfaces")}
- Traffic Ingress: {metrics.get("in_traffic_mbps", 0.0)} Mbps
- Traffic Egress: {metrics.get("out_traffic_mbps", 0.0)} Mbps
- Errors (Input/Output): In: {metrics.get("in_errors", 0)}, Out: {metrics.get("out_errors", 0)}
- Hardware Power Supply (PSU) Status Code: {metrics.get("psu_status", 1)}
- Hardware Fan Status Code: {metrics.get("fan_status", 1)}
-----------------------------------------

CRITICAL RULES FOR ANALYSIS:
1. Address the user's specific query intent directly in your "root_cause" and "recommendation".
2. Do not assume the model of the switch; refer to it simply as '{device}' or dynamically evaluate its ports.
3. Establish the "status" field based on your Focus Objective. Baseline rule: {default_status}.
4. If the user is asking about traffic, do not complain about shutdown ports as a 'CRITICAL' problem unless it's a known uplink.
5. Return ONLY a valid JSON object. No markdown, no prose, no code block backticks.

Required Output Format:
{{
  "status": "NORMAL|WARNING|CRITICAL",
  "root_cause": "Clear, concise answer addressing the user's query directly.",
  "recommendation": "Step-by-step logical action plan.",
  "confidence": 0.0-1.0
}}
"""
    return run_ai(prompt)
