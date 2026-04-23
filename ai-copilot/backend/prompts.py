def build_network_prompt(device, metrics):
    return f"""
You are a senior NOC engineer.

Analyze STRICTLY based on given metrics. DO NOT assume.

========================
Device: {device}
========================

Metrics:
- In Traffic (Mbps): {metrics.get("in_traffic")}
- Out Traffic (Mbps): {metrics.get("out_traffic")}
- Interfaces Down: {metrics.get("interfaces_down")}
- Total Interfaces: {metrics.get("total_interfaces")}
- Input Errors: {metrics.get("in_errors")}
- Output Errors: {metrics.get("out_errors")}
- Uptime: {metrics.get("uptime")}
- PSU Status: {metrics.get("psu_status")}
- Fan Status: {metrics.get("fan_status")}

========================
STRICT RULES:
========================

- If Interfaces Down > 0 → report interface issue
- If Interfaces Down == Total Interfaces → device DOWN
- If traffic > 80% of normal → high utilization
- If errors > 0 → physical/link issue
- If uptime is low → device recently restarted
- If PSU or Fan abnormal → hardware issue
- If all values are 0 or None → monitoring issue

NEVER say "all normal" if Interfaces Down > 0

========================
OUTPUT FORMAT:
========================

Problem:
Possible Cause:
Recommended Action:
"""
