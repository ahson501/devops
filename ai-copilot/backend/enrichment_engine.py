def classify_interface(iface):

    name = iface.get("interface", "")
    alias = (iface.get("alias") or "").lower()

    if "stack" in name.lower():
        return "stack"

    if name.startswith("Vl"):
        return "svi"

    if "trunk" in alias:
        return "uplink"

    if "ap" in alias:
        return "wireless"

    if "server" in alias:
        return "server"

    if alias == "":
        return "unused"

    return "access"
