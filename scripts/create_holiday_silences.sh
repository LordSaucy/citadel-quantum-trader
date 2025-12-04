#!/usr/bin/env bash
# =============================================================================
# create_holiday_silences.sh
#
# Purpose
# -------
#   - Pull the public holiday calendar from Investing.com (or any other API).
#   - Insert each holiday as a *silence* entry into the CQT `silence_periods`
#     table so the trading engine automatically skips those dates.
#   - Run once a day (cron or systemd timer) – the script is idempotent;
#     duplicate holidays are ignored thanks to a unique constraint.
#
# Expected table schema (PostgreSQL):
#
#   CREATE TABLE IF NOT EXISTS silence_periods (
#       id          SERIAL PRIMARY KEY,
#       start_ts    TIMESTAMPTZ NOT NULL,
#       end_ts      TIMESTAMPTZ NOT NULL,
#       reason      TEXT        NOT NULL,
#       source      TEXT        NOT NULL,
#       UNIQUE (start_ts, end_ts, source)
#   );
#
# Environment
# -----------
#   POSTGRES_URL – full connection string, e.g.
#       postgresql://cqt_user:secret@cqt-db:5432/cqt_ledger
#
#   HOLIDAY_API_URL – endpoint that returns a JSON array of holidays.
#       Default is Investing.com’s public‑holiday endpoint.
#
# Dependencies
# ------------
#   - curl
#   - jq
#   - psql (client, already present in the Docker image)
#
# =============================================================================

set -euo pipefail

# -------------------------------------------------------------------------
# Configuration (can be overridden via environment variables)
# -------------------------------------------------------------------------
POSTGRES_URL="${POSTGRES_URL:?POSTGRES_URL environment variable required}"
HOLIDAY_API_URL="${HOLIDAY_API_URL:-https://api.investing.com/api/holidays}"
USER_AGENT="CitadelBot/1.0 (+https://cqt.example.com)"

# -------------------------------------------------------------------------
# Helper: fetch the raw JSON payload
# -------------------------------------------------------------------------
fetch_holidays() {
    echo "Fetching holidays from ${HOLIDAY_API_URL} …" >&2
    curl -sSL -A "${USER_AGENT}" "${HOLIDAY_API_URL}" |
        jq -r '.data[] | @base64'   # encode each object for safe iteration
}

# -------------------------------------------------------------------------
# Helper: decode a base64‑encoded JSON object and extract fields
# -------------------------------------------------------------------------
decode_holiday() {
    local b64="$1"
    # Decode and extract fields; adjust field names if the API changes
    echo "${b64}" | base64 -d |
        jq -r '
            {
                title:   .title,
                date:    .date,          # format: YYYY-MM-DD
                country: .country // "",
                impact:  .impact // "high"
            }'
}

# -------------------------------------------------------------------------
# Insert a single holiday into the DB (UPSERT‑style)
# -------------------------------------------------------------------------
insert_holiday() {
    local title="$1"
    local iso_date="$2"
    local country="$3"
    local impact="$4"

    # For a full‑day silence we set start at 00:00:00 and end at 23:59:59 UTC
    local start_ts="${iso_date}T00:00:00+00:00"
    local end_ts="${iso_date}T23:59:59+00:00"

    # Build the INSERT … ON CONFLICT statement
    local sql="
        INSERT INTO silence_periods
            (start_ts, end_ts, reason, source)
        VALUES
            ('$start_ts'::timestamptz,
             '$end_ts'::timestamptz,
             '${title} (Holiday – ${country})',
             'investing.com')
        ON CONFLICT (start_ts, end_ts, source) DO NOTHING;
    "

    # Execute via psql (quiet mode, no password prompts)
    PGPASSWORD=$(echo "${POSTGRES_URL}" | sed -E 's#^postgresql://[^:]+:([^@]+)@.*#\1#') \
        psql "${POSTGRES_URL}" -c "${sql}" >/dev/null
}

# -------------------------------------------------------------------------
# Main driver
# -------------------------------------------------------------------------
main() {
    local count=0
    while IFS= read -r enc; do
        # Decode the holiday JSON object
        eval "$(decode_holiday "${enc}" | \
            jq -r '"title=\(.title) iso_date=\(.date) country=\(.country) impact=\(.impact)"')"

        # Skip if any mandatory field is missing (defensive)
        if [[ -z "${title:-}" || -z "${iso_date:-}" ]]; then
            echo "WARN: Skipping malformed entry: ${enc}" >&2
            continue
        fi

        insert_holiday "${title}" "${iso_date}" "${country}" "${impact}"
        ((count++))
    done < <(fetch_holidays)

    echo "✅ Inserted ${count} holiday silences (duplicates ignored)."
}

# -------------------------------------------------------------------------
# Execute
# -------------------------------------------------------------------------
main
