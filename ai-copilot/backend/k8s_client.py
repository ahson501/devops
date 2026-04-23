from kubernetes import client, config
import logging
import os

NAMESPACE = os.getenv("NAMESPACE", "monitoring")

try:
    config.load_incluster_config()
    logging.info("Loaded in-cluster config")
except Exception:
    logging.warning("Falling back to kubeconfig")
    config.load_kube_config()

v1 = client.CoreV1Api()


def get_devices():
    try:

        services = v1.list_namespaced_service(namespace=NAMESPACE)

        devices = []

        for svc in services.items:
            labels = svc.metadata.labels or {}
            if labels.get("snmp-target") == "true" or "snmp" in svc.metadata.name:
                devices.append({
                    "name": svc.metadata.name,
                    "type": labels.get("device_type", "unknown"),
                    "location": labels.get("location", "unknown")
                })

        return devices

    except Exception as e:
        logging.error(f"Failed to fetch devices: {str(e)}")
        return []

def trigger_healing_action(device, action="none"):
    try:
        # placeholder for future AI healing logic
        return {
            "device": device,
            "action": action,
            "status": "not_implemented"
        }
    except Exception as e:
        logging.error(f"Healing action failed: {str(e)}")
        return {"status": "error"}
