#!/usr/bin/env python3
"""
capacity_bench.py

Runs a configurable number of Citadel bucket containers,
samples host‑wide CPU% and RAM% while they are alive,
and returns a JSON array of per‑run samples.
"""
import argparse, json, os, subprocess, time
from datetime import datetime, timezone

import psutil

DOCKER_IMAGE = os.getenv("CITADEL_IMAGE", "citadel/trader:latest")
RUN_SECONDS = int(os.getenv("BENCH_RUN_SECONDS", "30"))   # how long each bucket lives
SAMPLES_PER_RUN = int(os.getenv("BENCH_SAMPLES", "5"))   # repetitions per bucket count


def launch_bucket(name: str):
    """Start a detached container with a unique name."""
    subprocess.run(
        ["docker", "run", "-d", "--name", name, DOCKER_IMAGE],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def stop_and_remove(name: str):
    subprocess.run(["docker", "stop", name], stdout=subprocess.DEVNULL)
    subprocess.run(["docker", "rm", name], stdout=subprocess.DEVNULL)


def sample_resources():
    """Return (cpu_percent, mem_percent) sampled over a 1‑second interval."""
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory().percent
    return cpu, mem


def bench_one_iteration(bucket_name: str):
    launch_bucket(bucket_name)
    # Give the container a moment to start up (especially if it pulls images)
    time.sleep(2)

    # Sample while the bucket is alive
    cpu, mem = sample_resources()
    stop_and_remove(bucket_name)

    return {"bucket": bucket_name, "cpu": cpu, "mem": mem,
            "ts": datetime.now(timezone.utc).isoformat()}


def bench_multi(count: int):
    """Run `count` buckets *concurrently* for the same duration."""
    # Spin up containers
    names = [f"bench_{count}_{i}" for i in range(count)]
    for n in names:
        launch_bucket(n)

    # Sleep while they all run
    time.sleep(RUN_SECONDS)

    # Sample once (host‑wide) – this approximates the *average* load per bucket
    cpu, mem = sample_resources()

    # Tear down
    for n in names:
        stop_and_remove(n)

    # Derive per‑bucket numbers (simple division)
    per_cpu = cpu / count
    per_mem = mem / count
    return {"concurrent": count, "cpu_total": cpu, "mem_total": mem,
            "cpu_per_bucket": per_cpu, "mem_per_bucket": per_mem,
            "ts": datetime.now(timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser(description="Citadel bucket capacity benchmark")
    parser.add_argument(
        "-c", "--counts", nargs="+", type=int, default=[1, 2, 4, 8, 16],
        help="Concurrent bucket counts to test"
    )
    args = parser.parse_args()

    results = []
    for cnt in args.counts:
        for _ in range(SAMPLES_PER_RUN):
            results.append(bench_multi(cnt))
            # small pause between runs to let the host settle
            time.sleep(3)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
