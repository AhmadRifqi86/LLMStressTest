#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  run.sh — Exponential OOM Stress Test for LLMs on Edge Devices
#
#  PURPOSE
#  ───────
#  Run every model through all seven benchmark suites with exponentially
#  increasing load (question count + context window) until OOM (kernel
#  OOM-killer crash) is forced on EACH benchmark.  The exact conditions
#  that trigger OOM are recorded per model per benchmark.
#
#  BENCHMARKS (run in order, each with exponential escalation)
#  ────────────────────────────────────────────────────────────
#  [1] ARC-Easy        0-shot MC     Clark et al. 2018  arXiv:1803.05457
#  [2] ARC-Challenge   0-shot MC     Clark et al. 2018  arXiv:1803.05457
#  [3] HellaSwag       0-shot MC     Zellers et al. 2019  arXiv:1905.07830
#  [4] MMLU            5-shot MC     Hendrycks et al. 2020  arXiv:2009.03300
#  [5] GSM8K           2-shot CoT    Cobbe et al. 2021  arXiv:2110.14168
#  [6] TruthfulQA      0-shot MC1    Lin et al. 2021  arXiv:2109.07958
#  [7] LongContext      needle@pos   Kamradt 2023; Liu et al. 2023  arXiv:2307.03172
#
#  ESCALATION MODEL
#  ─────────────────
#  Dataset size N is read from qcache.sh at runtime (not hard-coded).
#  For R = EXPONENTIAL_ROUNDS and each CTX_ESCALATION level:
#
#    base_q = ceil(N / 2^(R-1))
#    Round k: questions = min(N, base_q × 2^(k-1))
#
#  Example — N=100, R=6, ctx levels=(512 1024 2048 4096 8192):
#    Per ctx level: 4 → 8 → 16 → 32 → 64 → 100 questions
#    Total rounds until OOM: up to 30 (6 rounds × 5 ctx levels)
#    Each round is heavier: more tokens in prompt + larger KV cache.
#
#  LOGGING
#  ───────
#  Per-question lines:
#    Q42 [2026-03-30T14:23:01] [ctx=2048 round=14]: predicted=C correct=B ✗
#         latency=1234ms  cpu=87→92%  ram=1402→1418MB  prompt_tok=312
#
#  Resource timeseries CSV per model:
#    stress_oom_<TS>_<model>_resources.csv
#
#  USAGE
#  ─────
#    ./run.sh                             run all models
#    ./run.sh llama3.2-1b qwen3-0.6b     specific models only
#    ./run.sh --count 200                 200 questions per section
#    ./run.sh --count 200 llama3.2-1b    combine flags and names
#
# ═══════════════════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Source library modules in dependency order ────────────────────────
source "${SCRIPT_DIR}/lib/config.sh"
source "${SCRIPT_DIR}/lib/logging.sh"
source "${SCRIPT_DIR}/lib/resources.sh"
source "${SCRIPT_DIR}/lib/server.sh"
source "${SCRIPT_DIR}/lib/api.sh"
source "${SCRIPT_DIR}/lib/benchmarks.sh"
source "${SCRIPT_DIR}/lib/report.sh"

# ── Parse arguments ───────────────────────────────────────────────────
Q_COUNT=100
CONTINUE_MODE=0
SELECT_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --count)         Q_COUNT="$2"; shift 2 ;;
        --count=*)       Q_COUNT="${1#*=}"; shift ;;
        --continue|-c)   CONTINUE_MODE=1; shift ;;
        -*)              echo "Unknown flag: $1"; exit 1 ;;
        *)               SELECT_ARGS+=("$1"); shift ;;
    esac
done

# ── Build model list ──────────────────────────────────────────────────
if [ "${#SELECT_ARGS[@]}" -gt 0 ]; then
    MODELS=()
    for entry in "${ALL_MODELS[@]}"; do
        cfg=$(echo "$entry" | cut -d'|' -f3)
        for arg in "${SELECT_ARGS[@]}"; do
            [ "$cfg" = "$arg" ] && MODELS+=("$entry")
        done
    done
    if [ "${#MODELS[@]}" -eq 0 ]; then
        echo "ERROR: No models matched: ${SELECT_ARGS[*]}"
        echo "Available models:"
        for e in "${ALL_MODELS[@]}"; do
            echo "  $(echo "$e" | cut -d'|' -f3)"
        done
        exit 1
    fi
else
    MODELS=("${ALL_MODELS[@]}")
fi

# ── Continue mode: skip models already finished in the latest report ──
# A model is "done" when print_model_summary has written its block,
# which happens for both completed and oom@<bench> outcomes.
# The new report appends fresh results after the skipped ones.
if [ "$CONTINUE_MODE" -eq 1 ]; then
    # Find the most recent prior report (exclude the one we're about to write)
    RESUME_REPORT=$(ls -t stress_oom_*.txt 2>/dev/null \
        | grep -vx "$(basename "$REPORT_FILE")" | head -1 || true)

    if [ -z "$RESUME_REPORT" ]; then
        echo "ERROR: --continue requested but no previous stress_oom_*.txt found."
        echo "       Run without --continue first, then retry."
        exit 1
    fi

    echo "Continue mode — reading prior report: ${RESUME_REPORT}"

    # Extract model names whose summary block was written (completed or OOM'd)
    mapfile -t _DONE_MODELS < <(
        grep -oP '(?<=MODEL SUMMARY — ).*' "$RESUME_REPORT" 2>/dev/null \
        | awk '!seen[$0]++' || true
    )

    if [ "${#_DONE_MODELS[@]}" -gt 0 ]; then
        echo "  Already finished (${#_DONE_MODELS[@]}): ${_DONE_MODELS[*]}"
        FILTERED_MODELS=()
        for entry in "${MODELS[@]}"; do
            cfg=$(echo "$entry" | cut -d'|' -f3)
            _skip=0
            for done_cfg in "${_DONE_MODELS[@]}"; do
                [ "$cfg" = "$done_cfg" ] && _skip=1 && break
            done
            [ "$_skip" -eq 0 ] && FILTERED_MODELS+=("$entry")
        done
        MODELS=("${FILTERED_MODELS[@]}")
    fi

    if [ "${#MODELS[@]}" -eq 0 ]; then
        echo "Continue mode: nothing left to run — all models already finished."
        exit 0
    fi

    echo "  Remaining (${#MODELS[@]}): $(for e in "${MODELS[@]}"; do echo -n "$(echo "$e"|cut -d'|' -f3) "; done)"
    echo ""

    # Carry the prior report into the new one so the final table covers all models
    echo "# ── continued from ${RESUME_REPORT} ──" >> "$REPORT_FILE"
    cat "$RESUME_REPORT" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
fi

# ── Question cache ────────────────────────────────────────────────────
# Delete qcache.sh to force a refresh with a new --count value.
QCACHE="${SCRIPT_DIR}/qcache.sh"
if [ ! -f "$QCACHE" ]; then
    log "Question cache not found — fetching ${Q_COUNT} questions per section..."
    log "  Requires: pip install datasets"
    python3 "${SCRIPT_DIR}/data/fetch_questions.py" "$QCACHE" --count "$Q_COUNT"
    if [ ! -s "$QCACHE" ]; then
        echo "ERROR: fetch_questions.py produced an empty cache."
        echo "       Install datasets: pip install datasets"
        exit 1
    fi
    log "Questions cached → ${QCACHE}  (delete to refresh)"
fi

# shellcheck disable=SC1090
source "$QCACHE"
log "Loaded question cache:"
log "  ARC-Easy    : ${#ARC_E_Q[@]} questions"
log "  ARC-Challenge: ${#ARC_C_Q[@]} questions"
log "  HellaSwag   : ${#HS_Q[@]} questions"
log "  MMLU        : ${#MMLU_Q[@]} questions"
log "  GSM8K       : ${#GSM8K_Q[@]} questions"
log "  TruthfulQA  : ${#TQA_Q[@]} questions"

# Sanity check: all sections must have at least 1 question
for _arr in ARC_E_Q ARC_C_Q HS_Q MMLU_Q GSM8K_Q TQA_Q; do
    eval "_cnt=\${#${_arr}[@]}"
    if [ "${_cnt:-0}" -eq 0 ]; then
        echo "ERROR: ${_arr} is empty. Delete ${QCACHE} and re-run."
        exit 1
    fi
done

# ── Result stores ─────────────────────────────────────────────────────
declare -A R_STATUS R_TIME
declare -A R_OOM_BENCH R_OOM_ROUND R_OOM_CTX R_OOM_N_Q
declare -A R_SECTIONS_DONE
# R_CPU_AVG / R_CPU_PEAK / R_RAM_AVG / R_RAM_PEAK declared in resources.sh

ALL_TESTED_MODELS=()
BENCH_NAMES=("ARC-Easy" "ARC-Challenge" "HellaSwag" "MMLU" "GSM8K" "TruthfulQA" "LongCtx")

# ── Graceful exit trap ────────────────────────────────────────────────
trap 'stop_heartbeat 2>/dev/null; stop_resource_sampler 2>/dev/null
      echo ""
      echo "EXIT — last step: $(cat "$STEP_FILE" 2>/dev/null || echo "unknown")"' EXIT

# ── Header ────────────────────────────────────────────────────────────
sep2
log " EXPONENTIAL OOM STRESS TEST"
log " Started    : $(date '+%Y-%m-%d %H:%M:%S')"
log " Report     : ${REPORT_FILE}"
log " Target     : ${SSH_TARGET}"
log " Models     : ${#MODELS[@]}"
log " Benchmarks : ${BENCH_NAMES[*]}"
log " Q/section  : ${Q_COUNT} (actual from qcache.sh)"
log " Ctx levels : ${CTX_ESCALATION[*]}"
log " Rounds/ctx : ${EXPONENTIAL_ROUNDS}"
log " Goal       : Force OOM on every benchmark — record exact threshold"
sep2
log ""

BENCH_START=$(date +%s)
MDL_IDX=0

# ═════════════════════════════════════════════════════════════════════
#  Main model loop
# ═════════════════════════════════════════════════════════════════════
for model_entry in "${MODELS[@]}"; do
    MDL_IDX=$(( MDL_IDX + 1 ))
    HF_REPO=$(echo   "$model_entry" | cut -d'|' -f1)
    MODEL_FILE=$(echo "$model_entry" | cut -d'|' -f2)
    CURRENT_MODEL=$(echo "$model_entry" | cut -d'|' -f3)
    QUANT=$(echo "$MODEL_FILE" | grep -oE 'Q[0-9]+_[A-Z0-9_]+|Q[0-9]+_[0-9]+' | head -1 || echo "?")
    MDL_START=$(date +%s)
    sections_done=0

    ALL_TESTED_MODELS+=("$CURRENT_MODEL")

    # ── Liveness gate ─────────────────────────────────────────────────
    CURRENT_BENCH="pre_flight"; CURRENT_Q="liveness_check"; CURRENT_ROUND=0
    track_step "check_alive"
    if ! check_alive; then
        log "╔══════════════════════════════════════════════════════════════════╗"
        log "║  !! EDGE DEVICE UNREACHABLE — skipping model                    ║"
        log "$(printf '║  Model : %-57s║' "$CURRENT_MODEL")"
        log "$(printf '║  Target: %-57s║' "$SSH_TARGET")"
        log "╚══════════════════════════════════════════════════════════════════╝"
        R_STATUS["$CURRENT_MODEL"]="unreachable"
        R_OOM_BENCH["$CURRENT_MODEL"]="N/A"
        R_SECTIONS_DONE["$CURRENT_MODEL"]=0
        R_TIME["$CURRENT_MODEL"]=0
        continue
    fi

    clean_edge

    sep2
    log "[${MDL_IDX}/${#MODELS[@]}]  ${CURRENT_MODEL}  (${QUANT})"
    log "  File  : ${MODEL_FILE}"
    log "  Repo  : ${HF_REPO}"
    log "  Start : $(date '+%Y-%m-%d %H:%M:%S')"
    sep2

    # ── Transfer model to edge ────────────────────────────────────────
    CURRENT_BENCH="setup"; CURRENT_Q="transfer"
    if ! transfer_model "$HF_REPO" "$MODEL_FILE"; then
        R_STATUS["$CURRENT_MODEL"]="download_fail"
        R_OOM_BENCH["$CURRENT_MODEL"]="N/A"
        R_SECTIONS_DONE["$CURRENT_MODEL"]=0
        MDL_END=$(date +%s); R_TIME["$CURRENT_MODEL"]=$(( MDL_END - MDL_START ))
        continue
    fi

    # ── Start monitoring ──────────────────────────────────────────────
    start_heartbeat
    start_resource_sampler "$CURRENT_MODEL"
    log "  Resource sampler started → ${RESOURCE_CSV_FILE}"

    # ── Per-model OOM accumulators ────────────────────────────────────
    _oom_bench="none"; _oom_round=0; _oom_ctx=0; _oom_n_q=0

    # ── Helper: record OOM and finalize model ──────────────────────────
    _record_oom_and_skip() {
        local bench="$1"
        stop_heartbeat
        resource_output=$(stop_resource_sampler)
        R_STATUS["$CURRENT_MODEL"]="oom@${bench}"
        R_OOM_BENCH["$CURRENT_MODEL"]="$bench"
        R_OOM_ROUND["$CURRENT_MODEL"]="$BENCH_OOM_ROUND"
        R_OOM_CTX["$CURRENT_MODEL"]="$BENCH_OOM_CTX"
        R_OOM_N_Q["$CURRENT_MODEL"]="$BENCH_OOM_N_Q"
        R_SECTIONS_DONE["$CURRENT_MODEL"]="$sections_done"
        MDL_END=$(date +%s); R_TIME["$CURRENT_MODEL"]=$(( MDL_END - MDL_START ))

        sep
        log "  RESOURCE USAGE — ${CURRENT_MODEL}  (stopped at OOM)"
        log "$(printf '    %-24s │ %8s │ %9s │ %8s │ %9s │ %s' \
            'Benchmark/Ctx' 'CPUavg%' 'CPUpeak%' 'RAMavg' 'RAMpeak' 'Duration')"
        log "$(printf '    %-24s │ %8s │ %9s │ %8s │ %9s │ %s' \
            '────────────────────────' '────────' '─────────' '────────' '─────────' '────────')"
        parse_and_log_resources "$resource_output" "$CURRENT_MODEL"
        sep
        print_model_summary "$CURRENT_MODEL"

        clean_edge
    }

    # ═══════════════════════════════════════════════════════════════════
    #  [1] ARC-Easy
    # ═══════════════════════════════════════════════════════════════════
    CURRENT_BENCH="ARC-Easy"
    sep2
    log ""
    log "  ════ [1/7] ARC-Easy — 0-shot MC  [Clark et al. 2018] ════"
    run_benchmark_exponential \
        "ARC-Easy" ARC_E_Q ARC_E_ANS "$MC_SYSTEM_0SHOT" "mc" "$N_PREDICT_MC"
    sections_done=$(( sections_done + 1 ))

    if ! assert_alive; then
        _record_oom_and_skip "ARC-Easy"; continue
    fi

    # ═══════════════════════════════════════════════════════════════════
    #  [2] ARC-Challenge
    # ═══════════════════════════════════════════════════════════════════
    CURRENT_BENCH="ARC-Challenge"
    sep2
    log ""
    log "  ════ [2/7] ARC-Challenge — 0-shot MC  [Clark et al. 2018] ════"
    run_benchmark_exponential \
        "ARC-Challenge" ARC_C_Q ARC_C_ANS "$MC_SYSTEM_0SHOT" "mc" "$N_PREDICT_MC"
    sections_done=$(( sections_done + 1 ))

    if ! assert_alive; then
        _record_oom_and_skip "ARC-Challenge"; continue
    fi

    # ═══════════════════════════════════════════════════════════════════
    #  [3] HellaSwag
    # ═══════════════════════════════════════════════════════════════════
    CURRENT_BENCH="HellaSwag"
    sep2
    log ""
    log "  ════ [3/7] HellaSwag — 0-shot MC  [Zellers et al. 2019] ════"
    run_benchmark_exponential \
        "HellaSwag" HS_Q HS_ANS "$MC_SYSTEM_0SHOT" "mc" "$N_PREDICT_MC"
    sections_done=$(( sections_done + 1 ))

    if ! assert_alive; then
        _record_oom_and_skip "HellaSwag"; continue
    fi

    # ═══════════════════════════════════════════════════════════════════
    #  [4] MMLU  (5-shot system prompt)
    # ═══════════════════════════════════════════════════════════════════
    CURRENT_BENCH="MMLU"
    sep2
    log ""
    log "  ════ [4/7] MMLU — 5-shot MC  [Hendrycks et al. 2020] ════"
    run_benchmark_exponential \
        "MMLU" MMLU_Q MMLU_ANS "$MMLU_SYSTEM" "mc" "$N_PREDICT_MC"
    sections_done=$(( sections_done + 1 ))

    if ! assert_alive; then
        _record_oom_and_skip "MMLU"; continue
    fi

    # ═══════════════════════════════════════════════════════════════════
    #  [5] GSM8K  (2-shot chain-of-thought, 512 token budget)
    # ═══════════════════════════════════════════════════════════════════
    CURRENT_BENCH="GSM8K"
    sep2
    log ""
    log "  ════ [5/7] GSM8K — 2-shot CoT  [Cobbe et al. 2021] ════"
    run_benchmark_exponential \
        "GSM8K" GSM8K_Q GSM8K_ANS "" "gsm" "$N_PREDICT_MATH"
    sections_done=$(( sections_done + 1 ))

    if ! assert_alive; then
        _record_oom_and_skip "GSM8K"; continue
    fi

    # ═══════════════════════════════════════════════════════════════════
    #  [6] TruthfulQA
    # ═══════════════════════════════════════════════════════════════════
    CURRENT_BENCH="TruthfulQA"
    sep2
    log ""
    log "  ════ [6/7] TruthfulQA — 0-shot MC1  [Lin et al. 2021] ════"
    run_benchmark_exponential \
        "TruthfulQA" TQA_Q TQA_ANS "$MC_SYSTEM_0SHOT" "mc" "$N_PREDICT_MC"
    sections_done=$(( sections_done + 1 ))

    if ! assert_alive; then
        _record_oom_and_skip "TruthfulQA"; continue
    fi

    # ═══════════════════════════════════════════════════════════════════
    #  [7] Long Context / Needle-in-a-Haystack
    # ═══════════════════════════════════════════════════════════════════
    CURRENT_BENCH="LongCtx"
    sep2
    log ""
    log "  ════ [7/7] LongContext — needle retrieval  [Kamradt 2023; Liu et al. 2023] ════"
    run_long_context_exponential
    sections_done=$(( sections_done + 1 ))

    if ! assert_alive; then
        _record_oom_and_skip "LongCtx"; continue
    fi

    # ── Model survived all benchmarks ─────────────────────────────────
    stop_heartbeat
    resource_output=$(stop_resource_sampler)
    MDL_END=$(date +%s)
    R_TIME["$CURRENT_MODEL"]=$(( MDL_END - MDL_START ))
    R_STATUS["$CURRENT_MODEL"]="completed"
    R_OOM_BENCH["$CURRENT_MODEL"]="none"
    R_OOM_ROUND["$CURRENT_MODEL"]="—"
    R_OOM_CTX["$CURRENT_MODEL"]="—"
    R_OOM_N_Q["$CURRENT_MODEL"]="—"
    R_SECTIONS_DONE["$CURRENT_MODEL"]="$sections_done"

    sep
    log "  RESOURCE USAGE — ${CURRENT_MODEL}  (all benchmarks completed)"
    log "$(printf '    %-24s │ %8s │ %9s │ %8s │ %9s │ %s' \
        'Benchmark/Ctx' 'CPUavg%' 'CPUpeak%' 'RAMavg' 'RAMpeak' 'Duration')"
    log "$(printf '    %-24s │ %8s │ %9s │ %8s │ %9s │ %s' \
        '────────────────────────' '────────' '─────────' '────────' '─────────' '────────')"
    parse_and_log_resources "$resource_output" "$CURRENT_MODEL"
    sep
    print_model_summary "$CURRENT_MODEL"

    stop_server
    ssh "$SSH_TARGET" "rm -f ${REMOTE_MODEL_DIR}/${MODEL_FILE}" 2>/dev/null || true
    sleep 2
    log "  Cleanup done."
    log ""
done

# ═════════════════════════════════════════════════════════════════════
#  Final report
# ═════════════════════════════════════════════════════════════════════
BENCH_END=$(date +%s)

# In continue mode, re-read previously finished models from the prior
# report so the final table covers the full run, not just this session.
if [ "$CONTINUE_MODE" -eq 1 ] && [ -n "${RESUME_REPORT:-}" ]; then
    while IFS= read -r prev_model; do
        [ -z "$prev_model" ] && continue
        # Only add if not already in ALL_TESTED_MODELS (this session)
        _already=0
        for _m in "${ALL_TESTED_MODELS[@]}"; do
            [ "$_m" = "$prev_model" ] && _already=1 && break
        done
        if [ "$_already" -eq 0 ]; then
            ALL_TESTED_MODELS=("$prev_model" "${ALL_TESTED_MODELS[@]}")
            # Mark with a note — detailed stats stay in the prior report section
            R_STATUS["$prev_model"]="${R_STATUS[$prev_model]:-resumed}"
            R_OOM_BENCH["$prev_model"]="${R_OOM_BENCH[$prev_model]:-see-prior}"
            R_SECTIONS_DONE["$prev_model"]="${R_SECTIONS_DONE[$prev_model]:-?}"
            R_TIME["$prev_model"]="${R_TIME[$prev_model]:-0}"
        fi
    done < <(grep -oP '(?<=MODEL SUMMARY — ).*' "$RESUME_REPORT" 2>/dev/null \
             | awk '!seen[$0]++' || true)
fi

print_final_table $(( BENCH_END - BENCH_START ))
