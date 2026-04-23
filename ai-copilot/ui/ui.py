import streamlit as st
import requests
import pandas as pd
import os

# =========================
# CONFIG
# =========================

# This pulls the URL from the K8s Deployment YAML environment variable
BASE_URL = os.getenv("BACKEND_URL", "http://ai-copilot.monitoring.svc.cluster.local")

# Now your endpoints are built relative to that base
API_URL = f"{BASE_URL}/chat"
DEVICES_URL = f"{BASE_URL}/devices"

PROM_URL = "http://prometheus.monitoring.svc.cluster.local:9090/api/v1/query"
ALERT_URL = "http://alertmanager.monitoring.svc.cluster.local:9093/api/v2/alerts"

REFRESH_INTERVAL = 10

st.set_page_config(
    page_title="AICCBS NOC Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# STYLE
# =========================
st.markdown("""
<style>
body { background-color: #0e1117; color: white; }
.metric-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #1c1f26;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.title("🚀 ICCBS NOC Dashboard")
st.caption("Real-time Monitoring • AI Root Cause • Alerts • Topology View")

# =========================
# CACHE FUNCTIONS
# =========================

@st.cache_data(ttl=10)
def get_devices():
    try:
        res = requests.get(
            "http://ai-copilot.monitoring.svc.cluster.local/devices",
            timeout=5
        )
        return res.json().get("devices", [])
    except:
        return []


@st.cache_data(ttl=5)
def query_prometheus(query):
    try:
        res = requests.get(PROM_URL, params={"query": query}, timeout=10)
        data = res.json()

        if data.get("status") == "success" and data.get("data", {}).get("result"):
            try:
                return float(data["data"]["result"][0]["value"][1])
            except:
                return 0.0
        return 0.0
    except:
        return 0.0


@st.cache_data(ttl=5)
def get_alerts():
    try:
        res = requests.get(ALERT_URL, timeout=10)
        return res.json()
    except:
        return []

# =========================
# AI ANALYSIS
# =========================
def get_analysis(device, metrics, alerts):
    try:
        res = requests.post(
            API_URL,
            json={
                "device": device,
                "metrics": metrics,
                "alerts": alerts
            },
            timeout=30
        )
        return res.json()
    except Exception as e:
        return {"error": str(e)}

# =========================
# STATUS HELPER
# =========================
def get_status(text):
    t = str(text).lower()
    if "critical" in t or "down" in t:
        return "🔴 CRITICAL"
    elif "warning" in t or "high" in t:
        return "🟡 WARNING"
    return "🟢 NORMAL"

# =========================
# DEVICE DATA
# =========================
device_data = get_devices()
device_names = [d.get("name", "unknown") for d in device_data]

# =========================
# GROUP BY LOCATION (YOUR FEATURE FIXED)
# =========================
def group_devices(devices):
    grouped = {}
    for d in devices:
        loc = d.get("location", "unknown")
        name = d.get("name", "unnamed")
        grouped.setdefault(loc, []).append(name)
    return grouped

grouped_devices = group_devices(device_data)

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Settings")

devices = st.sidebar.multiselect(
    "Select Devices",
    device_names,
    default=device_names[:1] if device_names else []
)

auto_refresh = st.sidebar.checkbox("Auto Refresh")
analyze_btn = st.sidebar.button("🔍 Analyze")

# =========================
# MAIN
# =========================
if analyze_btn or auto_refresh:

    for device in devices:

        st.divider()
        st.subheader(f"📡 Device: {device}")

        # =========================
        # METRICS
        # =========================
        in_traffic = query_prometheus(
            f'irate(ifHCInOctets{{instance="{device}"}}[2m]) * 8 / 1024 / 1024'
        )

        out_traffic = query_prometheus(
            f'irate(ifHCOutOctets{{instance="{device}"}}[2m]) * 8 / 1024 / 1024'
        )

        interfaces_down = query_prometheus(
            f'ifOperStatus{{instance="{device}"}} == 2'
        )

        metrics = {
            "in_mbps": in_traffic,
            "out_mbps": out_traffic,
            "interfaces_down": interfaces_down
        }

        col1, col2, col3 = st.columns(3)

        col1.metric("📥 In Traffic", f"{in_traffic:.2f} Mbps")
        col2.metric("📤 Out Traffic", f"{out_traffic:.2f} Mbps")
        col3.metric("🔌 Interfaces Down", int(interfaces_down))

        # =========================
        # HISTORY (SAFE)
        # =========================
        if "history" not in st.session_state:
            st.session_state.history = {}

        if device not in st.session_state.history:
            st.session_state.history[device] = []

        st.session_state.history[device].append({
            "in": in_traffic,
            "out": out_traffic
        })

        st.session_state.history[device] = st.session_state.history[device][-20:]

        df = pd.DataFrame(st.session_state.history[device])
        st.line_chart(df)

        # =========================
        # ALERTS
        # =========================
        st.subheader("🚨 Active Alerts")

        alerts = get_alerts()

        if alerts:
            for alert in alerts[:5]:
                labels = alert.get("labels", {})
                name = labels.get("alertname", "N/A")
                severity = labels.get("severity", "unknown")

                if severity == "critical":
                    st.error(f"{name} ({severity})")
                elif severity == "warning":
                    st.warning(f"{name} ({severity})")
                else:
                    st.info(f"{name} ({severity})")
        else:
            st.success("No active alerts")

        # =========================
        # AI ANALYSIS
        # =========================
        st.subheader("🧠 AI Root Cause Analysis")

        with st.spinner("Analyzing..."):
            data = get_analysis(device, metrics, alerts)

        if "error" in data:
            st.error(data["error"])
            continue

        response = data.get("response", {})

        if isinstance(response, dict):

            status = get_status(response.get("status", ""))

            st.markdown(f"### Status: {status}")
            st.error(f"Root Cause: {response.get('root_cause', 'N/A')}")
            st.info(f"Recommendation: {response.get('recommendation', 'N/A')}")
            st.caption(f"Confidence: {response.get('confidence', 'N/A')}")

        else:
            st.code(str(response), language="markdown")

        st.caption(f"Model: {data.get('model_used', 'unknown')}")

# =========================
# TOPOLOGY VIEW (LOCATION GROUPING)
# =========================
st.sidebar.divider()
st.sidebar.subheader("📍 Site View")

for loc, devs in grouped_devices.items():
    with st.sidebar.expander(f"{loc} ({len(devs)})"):
        for d in devs:
            st.write("•", d)

# =========================
# AUTO REFRESH (PROPER STREAMLIT WAY)
# =========================
if auto_refresh:
    st.autorefresh(interval=REFRESH_INTERVAL * 1000, key="refresh")
