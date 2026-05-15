from fastapi import FastAPI, Request
from k8s_client import get_devices
from ai_engine import analyze_device, chat_handler

app = FastAPI()


@app.get("/")
def home():
    return {"status": "AI NOC Copilot Running"}


@app.get("/devices")
def list_devices():
    return {
        "devices": get_devices()
    }


@app.post("/analyze")
async def analyze(request: Request):
    data = await request.json()
    return analyze_device(data)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    return chat_handler(data)

# =========================
# SIMPLE ANALYZE ENDPOINT
# =========================
@app.get("/analyze/{device}")
def analyze_path(device: str):
    # This replaces the hardcoded rule-engine
    metrics = get_device_metrics(device)
    prompt = f"Perform a deep analysis of {device}. Data: {metrics}"
    return run_ai(prompt)
# =========================
# RECOMMENDATION ENDPOINT
# =========================
@app.get("/recommendation/{device}")
def recommend(device: str):
    metrics = get_device_metrics(device)
    prompt = f"Provide a specific technical recommendation for device {device} based on these metrics: {metrics}. Keep it concise."
    return run_ai(prompt)

