from fastapi import FastAPI, Request
import requests

from k8s_client import get_devices, trigger_healing_action
from prometheus_client import get_device_metrics
from ai_router import run_ai

app = FastAPI()

ALERT_URL = "http://alertmanager.monitoring.svc.cluster.local:9093/api/v2/alerts"


# =========================
# HEALTH
# =========================
@app.get("/")
def home():
    return {"status": "AI NOC Copilot Stage 4 Running"}

@app.get("/devices")
def list_devices():
    devices = get_devices()
    return {"devices": devices}

# =========================
# ALERTS
# =========================
def get_alerts():
    try:
        return requests.get(ALERT_URL, timeout=5).json()
    except:
        return []


# =========================
# ANOMALY SCORING
# =========================
def anomaly_score(metrics):
    if not metrics:
        return 100

    return min(
        (metrics.get("interfaces_down", 0) * 30) +
        (metrics.get("in_errors", 0) * 10) +
        (metrics.get("cpu", 0) / 2) +
        (metrics.get("memory", 0) / 2),
        100
    )


# =========================
# RULE ENGINE
# =========================
def pre_check(metrics):

    if not metrics:
        return {
            "status": "UNKNOWN",
            "root_cause": "No metrics available",
            "recommendation": "Check Prometheus/SNMP connectivity",
            "confidence": 0.5,
            "action": "none"
        }

    if metrics.get("interfaces_down", 0) == metrics.get("total_interfaces", 1):
        return {
            "status": "CRITICAL",
            "root_cause": "Device DOWN",
            "recommendation": "Check power/uplink",
            "confidence": 0.98,
            "action": "restart_device"
        }

    if metrics.get("interfaces_down", 0) > 0:
        return {
            "status": "WARNING",
            "root_cause": "Interface down detected",
            "recommendation": "Check interface/SFP",
            "confidence": 0.9,
            "action": "check_interfaces"
        }

    if metrics.get("in_errors", 0) > 100:
        return {
            "status": "WARNING",
            "root_cause": "High interface errors",
            "recommendation": "Check cable/duplex",
            "confidence": 0.85,
            "action": "check_interface_errors"
        }

    return None


# =========================
# PROMPT BUILDER
# =========================
def build_ai_prompt(device, metrics, alerts, score):

    return f"""
You are an Autonomous NOC AI.

DEVICE: {device}

METRICS:
{metrics}

ALERTS:
{alerts}

ANOMALY SCORE: {score}

Return ONLY JSON:
{{
  "status": "CRITICAL | WARNING | NORMAL",
  "root_cause": "...",
  "recommendation": "...",
  "confidence": 0.0-1.0,
  "action": "none | check | restart_interface | restart_pod | scale_up"
}}
"""

# =========================
# CHAT ENDPOINT
# =========================
@app.post("/chat")
async def chat(request: Request):

    try:
        data = await request.json()

        # ✅ Structured input (from UI)
        target_device = data.get("device")
        metrics = data.get("metrics", {})
        alerts = data.get("alerts", [])

        if not target_device:
            return {"error": "Device not provided"}

        # ✅ Filter alerts for this device (better matching)
        device_alerts = [
            a for a in alerts
            if target_device == a.get("labels", {}).get("instance")
        ]

        # =========================
        # RULE ENGINE FIRST
        # =========================
        quick = pre_check(metrics)

        if quick:
            if quick.get("action") != "none":
                trigger_healing_action(target_device, quick.get("action"))

            return {
                "response": quick,
                "model_used": "rule-engine",
                "anomaly_score": anomaly_score(metrics)
            }

        # =========================
        # AI ANALYSIS
        # =========================
        score = anomaly_score(metrics)

        prompt = build_ai_prompt(
            target_device,
            metrics,
            device_alerts,
            score
        )

        ai_response = run_ai(prompt)

        if "error" in ai_response:
            return {"error": ai_response["error"]}

        structured = ai_response.get("response")

        # =========================
        # SAFE JSON ENFORCEMENT
        # =========================
        if not isinstance(structured, dict):
            structured = {
                "status": "UNKNOWN",
                "root_cause": str(structured),
                "recommendation": "AI did not return valid JSON",
                "confidence": 0.5,
                "action": "none"
            }

        # =========================
        # AUTO HEALING
        # =========================
        if structured.get("status") == "CRITICAL":
            trigger_healing_action(
                target_device,
                structured.get("action", "none")
            )

        return {
            "response": structured,
            "model_used": ai_response.get("model_used", "ai"),
            "anomaly_score": score
        }

    except Exception as e:
        return {
            "error": str(e)
        }
