# src/aws/constants.py
import os

# -------------------------------------------------
# The AWS Account ID that *owns* all CQT buckets.
# You can set it in the environment (recommended) or hard‑code it.
# -------------------------------------------------
EXPECTED_BUCKET_OWNER = os.getenv(
    "CQT_EXPECTED_BUCKET_OWNER",
    "123456789012"  # <-- replace with your real AWS account ID
)
