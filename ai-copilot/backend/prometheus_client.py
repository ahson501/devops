import requests

PROM_URL = "http://prometheus-stack-kube-prom-prometheus.monitoring:9090/api/v1/query"


def query_prom(query):
    try:
        r = requests.get(PROM_URL, params={"query": query}, timeout=10)
        data = r.json()

        if data.get("status") != "success":
            return None

        result = data.get("data", {}).get("result", [])

        if not result:
            return 0

        # If multiple series, return sum of values
        values = [float(item["value"][1]) for item in result]

        return sum(values)

    except Exception as e:
        print("Prometheus query error:", e)
        return None

def get_device_metrics(device):
    metrics = {}

    # =========================
    # TRAFFIC (Mbps)
    # =========================
    metrics["in_traffic"] = query_prom(
        f'(sum(irate(ifHCInOctets{{service="{device}"}}[2m])) * 8) / 1024 / 1024'
    )

    metrics["out_traffic"] = query_prom(
        f'(sum(irate(ifHCOutOctets{{service="{device}"}}[2m])) * 8) / 1024 / 1024'
    )

    # =========================
    # INTERFACE DOWN COUNT
    # =========================
    metrics["interfaces_down"] = query_prom(
        f'count(ifOperStatus{{service="{device}"}} == 2)'
    )

    # =========================
    # TOTAL INTERFACES
    # =========================
    metrics["total_interfaces"] = query_prom(
        f'count(ifOperStatus{{service="{device}"}})'
    )

    # =========================
    # ERRORS
    # =========================
    metrics["in_errors"] = query_prom(
        f'sum(irate(ifInErrors{{service="{device}"}}[2m]))'
    )

    metrics["out_errors"] = query_prom(
        f'sum(irate(ifOutErrors{{service="{device}"}}[2m]))'
    )

    # =========================
    # DEVICE HEALTH
    # =========================
    metrics["uptime"] = query_prom(
        f'sysUpTime{{service="{device}"}} / 100'
    )

    # =========================
    # HARDWARE ALERTS
    # =========================
    metrics["psu_status"] = query_prom(
        f'max_over_time(ciscoEnvMonSupplyState{{service="{device}"}}[5m])'
    )

    metrics["fan_status"] = query_prom(
        f'max(ciscoEnvMonFanState{{service="{device}"}})'
    )

    return metrics
