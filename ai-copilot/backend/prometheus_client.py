import requests

PROM_URL = "http://prometheus-stack-kube-prom-prometheus.monitoring:9090/api/v1/query"


# =====================================================
# GENERIC PROM QUERY
# =====================================================
def query_prom(query):
    try:
        r = requests.get(
            PROM_URL,
            params={"query": query},
            timeout=10
        )

        data = r.json()

        if data.get("status") != "success":
            return 0

        result = data.get("data", {}).get("result", [])

        if not result:
            return 0

        values = []

        for item in result:
            try:
                values.append(float(item["value"][1]))
            except:
                continue

        if not values:
            return 0

        return round(sum(values), 2)

    except Exception as e:
        print(f"Prometheus query error: {e}")
        return 0


# =====================================================
# RAW PROM QUERY
# =====================================================
def raw_query_prom(query):
    try:
        r = requests.get(
            PROM_URL,
            params={"query": query},
            timeout=10
        )

        data = r.json()

        if data.get("status") != "success":
            return []

        return data.get("data", {}).get("result", [])

    except Exception as e:
        print(f"Raw Prometheus query error: {e}")
        return []


# =====================================================
# DOWN INTERFACES
# =====================================================
def get_down_interfaces(device):

    status_results = raw_query_prom(
        f'ifOperStatus{{service="{device}"}} == 2'
    )

    name_results = raw_query_prom(
        f'ifName{{service="{device}"}}'
    )

    alias_results = raw_query_prom(
        f'ifAlias{{service="{device}"}}'
    )

    # =====================================================
    # MAP ifIndex -> ifName
    # =====================================================
    name_map = {}

    for item in name_results:

        lbls = item.get("metric", {})

        idx = lbls.get("ifIndex")
        name = lbls.get("ifName")

        if idx and name:
            name_map[idx] = name

    # =====================================================
    # MAP ifIndex -> ifAlias
    # =====================================================
    alias_map = {}

    for item in alias_results:

        lbls = item.get("metric", {})

        idx = lbls.get("ifIndex")
        alias = lbls.get("ifAlias")

        if idx and alias:
            alias_map[idx] = alias

    # =====================================================
    # BUILD INTERFACE LIST
    # =====================================================
    interfaces = []

    for item in status_results:

        labels = item.get("metric", {})

        idx = labels.get("ifIndex", "")

        interface_name = name_map.get(idx) or f"ifIndex-{idx}"

        alias_val = alias_map.get(idx, "")

        interfaces.append({
            "interface": interface_name,
            "alias": alias_val,
            "description": alias_val,
            "ifIndex": idx
        })

    return interfaces


# =====================================================
# ERROR INTERFACES
# =====================================================
def get_error_interfaces(device):

    error_results = raw_query_prom(
        f'ifInErrors{{service="{device}"}} > 0'
    )

    name_results = raw_query_prom(
        f'ifName{{service="{device}"}}'
    )

    alias_results = raw_query_prom(
        f'ifAlias{{service="{device}"}}'
    )

    # =====================================================
    # MAP ifIndex -> ifName
    # =====================================================
    name_map = {}

    for item in name_results:

        lbls = item.get("metric", {})

        idx = lbls.get("ifIndex")
        name = lbls.get("ifName")

        if idx and name:
            name_map[idx] = name

    # =====================================================
    # MAP ifIndex -> ifAlias
    # =====================================================
    alias_map = {}

    for item in alias_results:

        lbls = item.get("metric", {})

        idx = lbls.get("ifIndex")
        alias = lbls.get("ifAlias")

        if idx and alias:
            alias_map[idx] = alias

    # =====================================================
    # BUILD ERROR INTERFACES
    # =====================================================
    interfaces = []

    for item in error_results:

        labels = item.get("metric", {})

        idx = labels.get("ifIndex", "")

        interface_name = name_map.get(idx) or f"ifIndex-{idx}"

        interfaces.append({
            "interface": interface_name,
            "alias": alias_map.get(idx, ""),
            "errors": item.get("value", [None, 0])[1]
            if len(item.get("value", [])) > 1 else 0
        })

    return interfaces


# =====================================================
# DEVICE METRICS
# =====================================================
def get_device_metrics(device):

    metrics = {}

    # =====================================================
    # DOWN INTERFACES
    # =====================================================
    down_interfaces = get_down_interfaces(device)

    # =====================================================
    # FILTER NON-PRODUCTION INTERFACES
    # =====================================================
    IGNORE_PREFIXES = [
        "Vl",
        "Stack",
        "Fa0",
        "Null",
        "Loopback"
    ]

    IGNORE_ALIASES = [
        "unused",
        "shutdown"
    ]

    filtered_interfaces = []

    for i in down_interfaces:

        name = i.get("interface", "")
        alias = (i.get("alias") or "").lower()

        # Ignore VLAN / Stack / Loopback / Null interfaces
        if any(name.startswith(p) for p in IGNORE_PREFIXES):
            continue

        # Ignore intentionally shutdown ports
        if any(x in alias for x in IGNORE_ALIASES):
            continue

        # Ignore empty access ports with no alias
        if not alias and name.startswith("Gi"):
            continue

        filtered_interfaces.append(i)

    down_interfaces = filtered_interfaces

    metrics["down_interfaces"] = down_interfaces
    metrics["interfaces_down"] = len(down_interfaces)

    # =====================================================
    # TOTAL INTERFACES
    # =====================================================
    metrics["total_interfaces"] = query_prom(
        f'count(ifOperStatus{{service="{device}"}})'
    )

    # =====================================================
    # TRAFFIC
    # =====================================================
    metrics["in_traffic_mbps"] = query_prom(
        f'(sum(irate(ifHCInOctets{{service="{device}"}}[2m])) * 8) / 1024 / 1024'
    )

    metrics["out_traffic_mbps"] = query_prom(
        f'(sum(irate(ifHCOutOctets{{service="{device}"}}[2m])) * 8) / 1024 / 1024'
    )

    # =====================================================
    # ERRORS
    # =====================================================
    metrics["in_errors"] = query_prom(
        f'sum(irate(ifInErrors{{service="{device}"}}[2m]))'
    )

    metrics["out_errors"] = query_prom(
        f'sum(irate(ifOutErrors{{service="{device}"}}[2m]))'
    )

    metrics["error_interfaces"] = get_error_interfaces(device)

    # =====================================================
    # UPTIME
    # =====================================================
    metrics["uptime_seconds"] = query_prom(
        f'sysUpTime{{service="{device}"}} / 100'
    )

    # =====================================================
    # HARDWARE HEALTH
    # =====================================================
    metrics["psu_status"] = query_prom(
        f'max(ciscoEnvMonSupplyState{{service="{device}"}})'
    )

    metrics["fan_status"] = query_prom(
        f'max(ciscoEnvMonFanState{{service="{device}"}})'
    )

    # =====================================================
    # MONITORING STATUS
    # =====================================================
    metrics["monitoring_status"] = "healthy"

    if (
        metrics["total_interfaces"] == 0 and
        metrics["in_traffic_mbps"] == 0 and
        metrics["out_traffic_mbps"] == 0
    ):
        metrics["monitoring_status"] = "no_snmp_data"

    return metrics
