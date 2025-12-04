def test_full_optimisation_cycle(docker_compose):
    # 1️⃣ Spin up optimiser + bot + Prometheus (already in compose)
    # 2️⃣ Wait for optimiser to finish (poll for new_config.yaml)
    import time, pathlib, subprocess
    cfg_path = pathlib.Path("./config/new_config.yaml")
    for _ in range(30):
        if cfg_path.is_file():
            break
        time.sleep(10)
    assert cfg_path.is_file(), "new_config.yaml never appeared"

    # 3️⃣ Verify bot reloaded (look for log line)
    logs = subprocess.check_output(["docker", "logs", "citadel-bot-1"]).decode()
    assert "[WATCHER] Config reloaded" in logs
