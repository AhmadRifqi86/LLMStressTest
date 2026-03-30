#!/usr/bin/env bash
# ── report.sh ─────────────────────────────────────────────────────────
# Final OOM summary table and per-model resource usage report.
# Reads from the R_* associative arrays populated by run.sh.

# ── Per-model inline summary (logged right after each model completes) ─
print_model_summary() {
    local model="$1"
    local status="${R_STATUS[$model]:-unknown}"
    local oom_bench="${R_OOM_BENCH[$model]:-none}"
    local oom_round="${R_OOM_ROUND[$model]:-—}"
    local oom_ctx="${R_OOM_CTX[$model]:-—}"
    local oom_n_q="${R_OOM_N_Q[$model]:-—}"
    local sections="${R_SECTIONS_DONE[$model]:-0}"
    local elapsed="${R_TIME[$model]:-0}"

    sep
    log "  MODEL SUMMARY — ${model}"
    log "    Status       : ${status}"
    log "    Sections done: ${sections} / ${#BENCH_NAMES[@]}"
    log "    Total time   : $(fmt_time "${elapsed:-0}")"
    if [ "$oom_bench" != "none" ] && [ -n "$oom_bench" ]; then
        log "    OOM benchmark: ${oom_bench}"
        log "    OOM round    : ${oom_round}"
        log "    OOM ctx size : ${oom_ctx} tokens"
        log "    OOM n_q      : ${oom_n_q}"
    else
        log "    OOM          : none — survived all rounds"
    fi
    sep
}

# ── Final cross-model OOM table ───────────────────────────────────────
print_final_table() {
    local total_secs="$1"

    sep2
    log ""
    log " EXPONENTIAL OOM STRESS TEST — FINAL REPORT"
    log " Run completed : $(date '+%Y-%m-%d %H:%M:%S')"
    log " Total runtime : $(fmt_time "${total_secs:-0}")"
    log " Report file   : ${REPORT_FILE}"
    log ""
    log " OOM column meanings:"
    log "   OOM-Bench  = benchmark that caused the crash"
    log "   OOM-Round  = global round number (doubling sequence across ctx levels)"
    log "   OOM-Ctx    = context window size (tokens) when OOM occurred"
    log "   OOM-Nq     = question / haystack-para count at the crashing round"
    log "   'none'     = model survived all rounds without OOM"
    sep2
    log ""

    local col="%-16s %-12s %-9s %-8s %-8s %-9s %-8s %-9s %-8s %-12s"
    log "$(printf "$col" \
        'Model' 'OOM-Bench' 'OOM-Round' 'OOM-Ctx' 'OOM-Nq' 'CPUpeak%' 'RAMpeak' 'Sections' 'Time' 'Status')"
    log "$(printf "$col" \
        '────────────────' '────────────' '─────────' '────────' '────────' '─────────' '────────' '─────────' '────────' '────────────')"

    for model in "${ALL_TESTED_MODELS[@]}"; do
        local status="${R_STATUS[$model]:-?}"
        local oom_bench="${R_OOM_BENCH[$model]:-none}"
        local oom_round="${R_OOM_ROUND[$model]:-N/A}"
        local oom_ctx="${R_OOM_CTX[$model]:-N/A}"
        local oom_n_q="${R_OOM_N_Q[$model]:-N/A}"
        local cpu_pk="${R_CPU_PEAK[$model]:-N/A}"
        local ram_pk="${R_RAM_PEAK[$model]:-N/A}"
        local sections="${R_SECTIONS_DONE[$model]:-0}/${#BENCH_NAMES[@]}"
        local elapsed="${R_TIME[$model]:-0}"

        [ "$cpu_pk" != "N/A" ] && cpu_pk="${cpu_pk}%"
        [ "$ram_pk" != "N/A" ] && ram_pk="${ram_pk}MB"
        [ "$oom_ctx" != "N/A" ] && [ "$oom_ctx" != "—" ] && oom_ctx="${oom_ctx}tok"

        log "$(printf "$col" \
            "$model" \
            "$oom_bench" \
            "$oom_round" \
            "$oom_ctx" \
            "$oom_n_q" \
            "$cpu_pk" \
            "$ram_pk" \
            "$sections" \
            "$(fmt_time "${elapsed:-0}")" \
            "$status")"
    done

    log ""

    # Count outcomes
    local n_oom=0 n_survived=0 n_failed=0
    for model in "${ALL_TESTED_MODELS[@]}"; do
        local s="${R_STATUS[$model]:-?}"
        case "$s" in
            oom@*)        n_oom=$(( n_oom + 1 )) ;;
            completed)    n_survived=$(( n_survived + 1 )) ;;
            *)            n_failed=$(( n_failed + 1 )) ;;
        esac
    done

    log "  Summary: ${n_oom} OOM crash(es)  |  ${n_survived} survived all rounds  |  ${n_failed} setup/infra failure(s)"
    log ""
    log "  Diagnostic: ssh ${SSH_TARGET} 'dmesg | grep -i oom | tail -10'"
    sep2
}
