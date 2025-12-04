import time, os, subprocess
from prometheus_api_client import PrometheusConnect

PROM = PrometheusConnect(url="http://prometheus:9090", disable_ssl=True)

def check_alert():
    alerts = PROM.get_current_alerts()
    for a in alerts:
        if a["labels"]["alertname"] == "ProductionExpectancyDrop":
            return True
    return False

while True:
    if check_alert():
        print("[WATCHER] Alert triggered – performing rollback")
        # Run the same commands as the manual rollback (inside the host)
        subprocess.run(["bash", "-c",
            "cd /opt/config && rm -f use_optimised_cfg.flag && git checkout HEAD~1 new_config.yaml && touch reload_now"],
            check=True)
        # Silence the alert for a minute so we don’t loop forever
        time.sleep(60)
    else:
        time.sleep(30)
