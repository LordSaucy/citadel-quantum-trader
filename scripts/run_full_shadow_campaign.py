#!/usr/bin/env python3
"""
run_full_shadow_campaign.py

Convenient wrapper that:
  1️⃣  Runs the paper‑trading campaign (baseline)
  2️⃣  Runs the shadow‑mode campaign (live‑mirror)
  3️⃣  Generates the side‑by‑side KPI report
"""

import subprocess
import sys
from pathlib import Path

def run_compose(compose_file: str, override_file: str, name: str) -> None:
    print(f"\n=== Starting {name} stack ===")
    subprocess.check_call(
        [
            "docker",
            "compose",
            "-f",
            "docker-compose.yml",
            "-f",
            compose_file,
            "up",
            "-d",
        ]
    )
    # Wait for health endpoint (re‑use the same helper from the orchestrator)
    from scripts.run_paper_trading import wait_for_bot
    if not wait_for_bot():
        raise RuntimeError(f"{name} never became healthy")
    print(f"{name} is healthy – let it run …")

def stop_compose(compose_file: str) -> None:
    subprocess.run(
        ["docker", "compose", "-f", "docker-compose.yml", "-f", compose_file, "down"],
        check=False,
    )

def main() -> int:
    # -----------------------------------------------------------------
    # 1️⃣  Paper run (baseline) – 48 h or 500 trades
    # -----------------------------------------------------------------
    run_compose("docker-compose.paper.yml", "docker-compose.paper.yml", "Paper")
    # Paper orchestrator will stop itself after the condition is met.
    # We just wait for the container to disappear.
    print("\nWaiting for Paper run to finish …")
    subprocess.check_call(
        ["docker", "wait", "citadel-bot"]  # block until container exits
    )
    stop_compose("docker-compose.paper.yml")

    # -----------------------------------------------------------------
    # 2️⃣  Shadow run (live‑mirror) – same duration
    # -----------------------------------------------------------------
    run_compose("docker-compose.shadow.yml", "docker-compose.shadow.yml", "Shadow")
    print("\nWaiting for Shadow run to finish …")
    subprocess.check_call(
        ["docker", "wait", "citadel-bot"]
    )
    stop_compose("docker-compose.shadow.yml")

    # -----------------------------------------------------------------
    # 3️⃣  Compare the two runs
    # -----------------------------------------------------------------
    print("\nGenerating side‑by‑side report …")
    subprocess.check_call(
        [
            "python",
            "scripts/compare_shadow_vs_paper.py",
            "--paper-csv",
            "paper_trades.csv",
            "--shadow-log",
            "shadow.log",
            "--output",
            "shadow_vs_paper.md",
        ]
    )
    print("\n✅ Full Shadow campaign completed – see shadow_vs_paper.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
Run it
chmod +x scripts/run_full_shadow_campaign.py
./scripts/run_full_shadow_campaign.py
