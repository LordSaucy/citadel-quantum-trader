#!/usr/bin/env bash
set -euo pipefail

# Export Redis connection for the Python process
export REDIS_HOST=$(yq e '.redis.host' config.yaml)
export REDIS_PORT=$(yq e '.redis.port' config.yaml)
export REDIS_DB=$(yq e '.redis.db' config.yaml)
