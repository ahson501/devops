def calculate_severity(enriched):

    if enriched["critical_uplinks_down"] > 0:
        return "CRITICAL"

    if enriched["wireless_trunks_down"] > 2:
        return "WARNING"

    if enriched["error_rate"] > 20:
        return "WARNING"

    return "NORMAL"
