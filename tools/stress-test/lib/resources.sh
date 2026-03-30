#!/usr/bin/env bash
# ── resources.sh ──────────────────────────────────────────────────────
# CPU/RAM background sampler, instantaneous snapshots, heartbeat, and
# OOM/crash detection via SSH liveness checks.

# ── State globals ─────────────────────────────────────────────────────
RESOURCE_SAMPLER_PID=""
RESOURCE_SAMPLE_FILE=""
RESOURCE_CSV_FILE=""
HEARTBEAT_PID=""

# Per-model resource results — populated by parse_and_log_resources()
declare -A R_CPU_AVG R_CPU_PEAK R_RAM_AVG R_RAM_PEAK

# ── Heartbeat ─────────────────────────────────────────────────────────
# Runs in background; logs to stderr when SSH is lost.
start_heartbeat() {
    (
        while true; do
            sleep 5
            if ! ssh -o ConnectTimeout=4 -o BatchMode=yes -o StrictHostKeyChecking=no \
                    "$SSH_TARGET" "echo hb" 2>/dev/null | grep -q "hb"; then
                printf '\n[%s] !! HEARTBEAT LOST (model=%s bench=%s round=%s) !!\n' \
                    "$(date '+%H:%M:%S')" \
                    "${CURRENT_MODEL:-none}" \
                    "${CURRENT_BENCH:-none}" \
                    "${CURRENT_ROUND:-0}" >&2
                printf '[%s] Last step: %s\n' \
                    "$(date '+%H:%M:%S')" \
                    "$(cat "$STEP_FILE" 2>/dev/null || echo "unknown")" >&2
                break
            fi
        done
    ) &
    HEARTBEAT_PID=$!
}

stop_heartbeat() {
    if [ -n "$HEARTBEAT_PID" ]; then
        kill "$HEARTBEAT_PID" 2>/dev/null || true
        wait "$HEARTBEAT_PID" 2>/dev/null || true
        HEARTBEAT_PID=""
    fi
}

# ── OOM / crash detection ─────────────────────────────────────────────

# check_alive: returns 0 if SSH responds, 1 otherwise
check_alive() {
    ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no \
        "$SSH_TARGET" "echo alive" 2>/dev/null | grep -q "alive"
}

# assert_alive: check_alive + detailed OOM banner to log on failure
assert_alive() {
    if ! check_alive; then
        local last; last=$(cat "$STEP_FILE" 2>/dev/null || echo "unknown")
        log ""
        log "╔══════════════════════════════════════════════════════════════════╗"
        log "║  !! OOM / CRASH DETECTED                                        ║"
        log "╠══════════════════════════════════════════════════════════════════╣"
        log "$(printf '║  Model  : %-54s║' "${CURRENT_MODEL:-?}")"
        log "$(printf '║  Bench  : %-54s║' "${CURRENT_BENCH:-?}")"
        log "$(printf '║  Round  : %-54s║' "round=${CURRENT_ROUND:-?} ctx=${CURRENT_CTX_SIZE:-?}")"
        log "$(printf '║  Q      : %-54s║' "${CURRENT_Q:-?}")"
        log "$(printf '║  Step   : %-54s║' "${last}")"
        log "╚══════════════════════════════════════════════════════════════════╝"
        return 1
    fi
    return 0
}

# ── Instantaneous resource snapshots ─────────────────────────────────

# measure_ram_mb: returns integer MB used RAM on edge device
measure_ram_mb() {
    ssh -o ConnectTimeout=3 -o BatchMode=yes -o StrictHostKeyChecking=no \
        "$SSH_TARGET" \
        "free -m | awk '/^Mem:/{print \$3}'" 2>/dev/null || echo "0"
}

# snapshot_resources: returns "cpu_pct ram_mb" in a single SSH round-trip
snapshot_resources() {
    ssh -o ConnectTimeout=3 -o BatchMode=yes -o StrictHostKeyChecking=no \
        "$SSH_TARGET" \
        "cpu=\$(ps -C llama-server -o %cpu= 2>/dev/null | awk '{s+=\$1}END{printf \"%.1f\",s+0}'); \
         ram=\$(free -m | awk '/^Mem:/{print \$3}'); \
         echo \"\${cpu:-0} \${ram:-0}\"" 2>/dev/null || echo "0 0"
}

# ── Background resource sampler ───────────────────────────────────────
# Polls llama-server CPU% and system RAM every 2 s.
# Writes raw samples to a temp file; stop_resource_sampler() processes them
# into a per-model CSV and emits stats to stdout for the caller to capture.

start_resource_sampler() {
    local model_name="$1"
    RESOURCE_SAMPLE_FILE=$(mktemp /tmp/stress_res_XXXXXX)
    RESOURCE_CSV_FILE="stress_oom_${TS}_${model_name}_resources.csv"
    echo "timestamp,elapsed_s,benchmark,round,ctx_size,cpu_pct,ram_used_mb" > "$RESOURCE_CSV_FILE"
    local t0; t0=$(date +%s)
    echo "T0=${t0}" >> "$RESOURCE_SAMPLE_FILE"
    (
        while true; do
            local line
            line=$(ssh -o ConnectTimeout=3 -o BatchMode=yes -o StrictHostKeyChecking=no \
                "$SSH_TARGET" \
                "cpu=\$(ps -C llama-server -o %cpu= 2>/dev/null | awk '{s+=\$1}END{printf \"%.1f\",s+0}'); \
                 ram=\$(free -m | awk '/^Mem:/{print \$3}'); \
                 echo \"\$(date +%s) \${cpu:-0} \${ram:-0}\"" \
                2>/dev/null || echo "")
            [ -n "$line" ] && echo "$line" >> "$RESOURCE_SAMPLE_FILE"
            sleep 2
        done
    ) &
    RESOURCE_SAMPLER_PID=$!
}

# mark_section: write a section boundary into the sample file so the
# post-processor can group samples by benchmark, round, and ctx size.
mark_section() {
    local label="${1:-unknown}"
    local rnd="${CURRENT_ROUND:-0}"
    local ctx="${CURRENT_CTX_SIZE:-0}"
    [ -n "$RESOURCE_SAMPLE_FILE" ] && \
        echo "MARK $(date +%s) ${label} ${rnd} ${ctx}" >> "$RESOURCE_SAMPLE_FILE"
}

# stop_resource_sampler: kill background sampler, run Python post-processor
# to build the CSV and compute per-bench + overall stats.
# Emits key=value lines and BENCH| lines to stdout (captured by caller).
stop_resource_sampler() {
    if [ -n "$RESOURCE_SAMPLER_PID" ]; then
        kill "$RESOURCE_SAMPLER_PID" 2>/dev/null || true
        wait "$RESOURCE_SAMPLER_PID" 2>/dev/null || true
        RESOURCE_SAMPLER_PID=""
    fi
    [ ! -f "${RESOURCE_SAMPLE_FILE:-}" ] && return

    python3 - "$RESOURCE_SAMPLE_FILE" "$RESOURCE_CSV_FILE" << 'PYEOF'
import sys

sample_file = sys.argv[1]
csv_file    = sys.argv[2]

lines  = open(sample_file).readlines()
t0     = None
data   = []    # (epoch, cpu_pct, ram_mb)
marks  = []    # (epoch, bench_label, round_str, ctx_str)

for line in lines:
    line = line.strip()
    if line.startswith('T0='):
        t0 = int(line[3:])
    elif line.startswith('MARK'):
        parts = line.split(None, 4)
        bench = parts[2] if len(parts) > 2 else 'unknown'
        rnd   = parts[3] if len(parts) > 3 else '0'
        ctx   = parts[4] if len(parts) > 4 else '0'
        marks.append((int(parts[1]), bench, rnd, ctx))
    else:
        parts = line.split()
        if len(parts) == 3:
            try:
                data.append((int(parts[0]), float(parts[1]), int(parts[2])))
            except ValueError:
                pass

if not t0:
    t0 = data[0][0] if data else 0

def section_for(ts):
    bench, rnd, ctx = 'init', '0', '0'
    for mk_ts, mk_bench, mk_round, mk_ctx in marks:
        if ts >= mk_ts:
            bench, rnd, ctx = mk_bench, mk_round, mk_ctx
    return bench, rnd, ctx

# Write timeseries CSV
with open(csv_file, 'a') as f:
    for epoch, cpu, ram in data:
        elapsed = epoch - t0
        bench, rnd, ctx = section_for(epoch)
        f.write(f"{epoch},{elapsed},{bench},{rnd},{ctx},{cpu:.1f},{ram}\n")

# Overall stats
if data:
    cpus = [d[1] for d in data]
    rams = [d[2] for d in data]
    print(f"OVERALL_CPU_AVG={sum(cpus)/len(cpus):.1f}")
    print(f"OVERALL_CPU_PEAK={max(cpus):.1f}")
    print(f"OVERALL_RAM_AVG={int(sum(rams)/len(rams))}")
    print(f"OVERALL_RAM_PEAK={max(rams)}")
else:
    print("OVERALL_CPU_AVG=N/A")
    print("OVERALL_CPU_PEAK=N/A")
    print("OVERALL_RAM_AVG=N/A")
    print("OVERALL_RAM_PEAK=N/A")

# Per-benchmark stats grouped by "bench/ctxN" key
bench_data = {}
for epoch, cpu, ram in data:
    bench, rnd, ctx = section_for(epoch)
    key = f"{bench}/ctx{ctx}"
    bench_data.setdefault(key, {'cpu': [], 'ram': [], 'ts': []})
    bench_data[key]['cpu'].append(cpu)
    bench_data[key]['ram'].append(ram)
    bench_data[key]['ts'].append(epoch)

# Emit in order of first appearance
seen = []
for epoch, cpu, ram in data:
    bench, rnd, ctx = section_for(epoch)
    key = f"{bench}/ctx{ctx}"
    if key not in seen:
        seen.append(key)

for key in seen:
    d = bench_data[key]
    cpu_avg = sum(d['cpu']) / len(d['cpu']) if d['cpu'] else 0
    cpu_pk  = max(d['cpu']) if d['cpu'] else 0
    ram_avg = int(sum(d['ram']) / len(d['ram'])) if d['ram'] else 0
    ram_pk  = max(d['ram']) if d['ram'] else 0
    dur     = d['ts'][-1] - d['ts'][0] if len(d['ts']) > 1 else 0
    print(f"BENCH|{key}|{cpu_avg:.1f}|{cpu_pk:.1f}|{ram_avg}|{ram_pk}|{dur}")
PYEOF

    rm -f "$RESOURCE_SAMPLE_FILE"
    RESOURCE_SAMPLE_FILE=""
}

# parse_and_log_resources: parse the output of stop_resource_sampler,
# log a formatted per-benchmark table, and populate R_CPU/RAM globals.
parse_and_log_resources() {
    local raw_output="$1"
    local model="$2"

    local cpu_avg="N/A" cpu_peak="N/A" ram_avg="N/A" ram_peak="N/A"

    while IFS= read -r line; do
        case "$line" in
            OVERALL_CPU_AVG=*)  cpu_avg="${line#*=}"  ;;
            OVERALL_CPU_PEAK=*) cpu_peak="${line#*=}" ;;
            OVERALL_RAM_AVG=*)  ram_avg="${line#*=}"  ;;
            OVERALL_RAM_PEAK=*) ram_peak="${line#*=}" ;;
            BENCH|*)
                IFS='|' read -r _ bench_key b_cpu_a b_cpu_p b_ram_a b_ram_p b_dur <<< "$line"
                local dur_fmt; dur_fmt=$(fmt_time "${b_dur:-0}")
                log "$(printf '    %-24s │ %7s%% │ %8s%% │ %7s MB │ %8s MB │ %s' \
                    "$bench_key" "$b_cpu_a" "$b_cpu_p" "$b_ram_a" "$b_ram_p" "$dur_fmt")"
                ;;
        esac
    done <<< "$raw_output"

    log "$(printf '    %-24s │ %7s%% │ %8s%% │ %7s MB │ %8s MB │ %s' \
        'OVERALL' "$cpu_avg" "$cpu_peak" "$ram_avg" "$ram_peak" "—")"

    R_CPU_AVG["$model"]="$cpu_avg"
    R_CPU_PEAK["$model"]="$cpu_peak"
    R_RAM_AVG["$model"]="$ram_avg"
    R_RAM_PEAK["$model"]="$ram_peak"

    log "  Resource timeseries CSV → ${RESOURCE_CSV_FILE}"
}
