#!/usr/bin/env bash
# ── logging.sh ────────────────────────────────────────────────────────
# Timestamped logging, report file management, and progress tracking.
# All output goes to both stdout and REPORT_FILE simultaneously.

# ── Report file (set before sourcing if you want a custom name) ───────
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
REPORT_FILE="${REPORT_FILE:-stress_oom_${TS}.txt}"
STEP_FILE="stress_oom_laststep.txt"

# ── Core log functions ────────────────────────────────────────────────

# log: plain message to stdout + report file
log() {
    local msg="$*"
    echo "$msg"
    echo "$msg" >> "$REPORT_FILE"
}

# log_ts: same as log but prepends ISO timestamp
log_ts() {
    local ts; ts=$(date '+%Y-%m-%d %H:%M:%S.%3N')
    log "[${ts}] $*"
}

# Dividers
sep()  { log "────────────────────────────────────────────────────────────────────"; }
sep2() { log "════════════════════════════════════════════════════════════════════"; }

# ── Timestamp helpers ─────────────────────────────────────────────────

# iso_ts: ISO-8601 local timestamp (no milliseconds) — used in per-question lines
iso_ts() { date '+%Y-%m-%dT%H:%M:%S'; }

# ms_ts: Unix epoch in milliseconds — used for latency measurements
ms_ts()  { date +%s%3N; }

# fmt_time: format integer seconds as "Xm00s"
fmt_time() {
    local secs="${1:-0}"
    printf "%dm%02ds" $(( secs / 60 )) $(( secs % 60 ))
}

# ── Progress step tracker ─────────────────────────────────────────────
# Writes a one-line "breadcrumb" to STEP_FILE so that EXIT traps can
# report exactly where the run was when a crash or interrupt happened.
track_step() {
    printf '[%s] model=%s bench=%s round=%s q=%s step=%s\n' \
        "$(date '+%H:%M:%S')" \
        "${CURRENT_MODEL:-none}" \
        "${CURRENT_BENCH:-none}" \
        "${CURRENT_ROUND:-0}" \
        "${CURRENT_Q:-none}" \
        "$1" > "$STEP_FILE"
}
