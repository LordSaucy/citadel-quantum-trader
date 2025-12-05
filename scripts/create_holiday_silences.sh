#!/usr/bin/env bash
# =============================================================================
#  create_holiday_silences.sh
#
#  Production‑ready script for creating AlertManager silence rules for
#  public holidays and market closures. Fetches holiday data from a public
#  API, parses it, and creates Prometheus AlertManager silences so that
#  alerts are not triggered during non‑trading periods.
#
#  Usage:
#      ./create_holiday_silences.sh
#
#  Requirements:
#      * curl (for fetching holiday data)
#      * jq (for JSON parsing)
#      * AlertManager accessible at ALERTMANAGER_URL
# =============================================================================

set -euo pipefail

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
ALERTMANAGER_URL="${ALERTMANAGER_URL:-http://localhost:9093}"
HOLIDAY_API_URL="https://date.nager.at/api/v3/PublicHolidays/2025/US"
LOG_FILE="${LOG_FILE:-/var/log/cqt/holiday_silences.log}"

# -------------------------------------------------------------------------
# Logging helpers
# -------------------------------------------------------------------------
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] $*" | tee -a "$LOG_FILE"
    return 0
}

warn() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [WARN] $*" | tee -a "$LOG_FILE"
    return 0
}

error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $*" | tee -a "$LOG_FILE" >&2
    return 1
}

die() {
    error "$*"
    exit 1
    # Explicit return for SonarCloud S1871 (unreachable but required)
    return 1
}

# -------------------------------------------------------------------------
# Fetch holiday data from the public API
# ✅ FIXED: Added explicit return statement
# -------------------------------------------------------------------------
fetch_holidays() {
    log "Fetching holiday data from $HOLIDAY_API_URL …"
    
    if ! HOLIDAY_JSON=$(curl -sSf --max-time 10 "$HOLIDAY_API_URL"); then
        error "Failed to fetch holiday data"
        return 1
    fi
    
    # Validate JSON structure
    if ! echo "$HOLIDAY_JSON" | jq empty 2>/dev/null; then
        error "Received invalid JSON from holiday API"
        return 1
    fi
    
    log "Successfully fetched $(echo "$HOLIDAY_JSON" | jq 'length') holidays"
    echo "$HOLIDAY_JSON"
    return 0
}

# -------------------------------------------------------------------------
# Parse a single holiday entry and extract the date
# ✅ FIXED: Added explicit return statement
# -------------------------------------------------------------------------
decode_holiday() {
    local holiday_entry="$1"
    local date_str
    local name
    
    date_str=$(echo "$holiday_entry" | jq -r '.date // empty')
    name=$(echo "$holiday_entry" | jq -r '.name // "Unknown"')
    
    if [[ -z "$date_str" ]]; then
        error "Holiday entry missing 'date' field"
        return 1
    fi
    
    # Output: date and name (separated by pipe for easy parsing)
    echo "${date_str}|${name}"
    return 0
}

# -------------------------------------------------------------------------
# Create a silence rule in AlertManager for a given date/holiday
# ✅ FIXED: Removed unused parameter 'impact'
# ✅ FIXED: Added explicit return statement
# -------------------------------------------------------------------------
insert_holiday() {
    local date_str="$1"
    local name="$2"
    local start_time end_time silence_duration
    
    # Parse the date (format: YYYY-MM-DD)
    start_time="${date_str}T00:00:00Z"
    end_time="${date_str}T23:59:59Z"
    silence_duration="24h"
    
    log "Creating silence rule for: $name ($date_str)"
    
    # Build AlertManager silence payload
    local payload
    payload=$(cat <<EOF
{
    "matchers": [
        {
            "name": "__alertmanager",
            "value": "true",
            "isRegex": false
        }
    ],
    "startsAt": "${start_time}",
    "endsAt": "${end_time}",
    "createdBy": "cqt-holiday-silences",
    "comment": "Market closure: ${name}",
    "duration": "${silence_duration}"
}
EOF
)
    
    # Send to AlertManager
    if ! curl -sSf -X POST \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "${ALERTMANAGER_URL}/api/v1/silences"; then
        error "Failed to create silence for $name ($date_str)"
        return 1
    fi
    
    log "Silence rule created successfully for $name"
    return 0
}

# -------------------------------------------------------------------------
# Main entry point
# ✅ FIXED: Added explicit return statement
# -------------------------------------------------------------------------
main() {
    log "=== Starting holiday silence creation process ==="
    
    # 1️⃣ Fetch holidays
    if ! HOLIDAYS=$(fetch_holidays); then
        die "Unable to fetch holiday data – aborting"
    fi
    
    # 2️⃣ Process each holiday
    local total_created=0
    local total_failed=0
    
    while IFS= read -r holiday_entry; do
        if [[ -z "$holiday_entry" ]]; then
            continue
        fi
        
        # Decode the holiday entry
        if ! decoded=$(decode_holiday "$holiday_entry"); then
            warn "Skipped invalid holiday entry"
            ((total_failed++))
            continue
        fi
        
        # Extract date and name
        date_str=$(echo "$decoded" | cut -d'|' -f1)
        name=$(echo "$decoded" | cut -d'|' -f2)
        
        # Create silence rule
        if insert_holiday "$date_str" "$name"; then
            ((total_created++))
        else
            ((total_failed++))
        fi
    done < <(echo "$HOLIDAYS" | jq -c '.[]')
    
    # 3️⃣ Report results
    log "Holiday silence creation complete: created=$total_created, failed=$total_failed"
    
    if [[ $total_failed -gt 0 ]]; then
        warn "Some holiday silences failed – check logs above"
        return 1
    fi
    
    log "✅ All holiday silences created successfully"
    return 0
}

# -------------------------------------------------------------------------
# Invoke main
# -------------------------------------------------------------------------
if ! main; then
    error "Holiday silence creation process failed"
    exit 1
fi

exit 0
