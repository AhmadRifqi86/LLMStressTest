#!/usr/bin/env bash
# ── server.sh ─────────────────────────────────────────────────────────
# llama-server lifecycle: start, stop, wait-for-ready, restart with a
# new context size, model transfer, and KV cache inspection.

# Set by run.sh before calling start_server / restart_server_with_ctx.
MODEL_FILE=""
CURRENT_CTX_SIZE=512

# ── Health / readiness ────────────────────────────────────────────────

wait_for_server() {
    local elapsed=0
    echo -n "    Waiting for llama-server"
    until curl -s "http://${SSH_HOST}:${SERVER_PORT}/health" 2>/dev/null | grep -q '"status":"ok"'; do
        sleep 2
        elapsed=$(( elapsed + 2 ))
        echo -n "."
        if [ "$elapsed" -ge "$SERVER_READY_TIMEOUT" ]; then
            echo " TIMEOUT"
            return 1
        fi
    done
    echo " ready (${elapsed}s)"
    return 0
}

# ── Stop ──────────────────────────────────────────────────────────────

stop_server() {
    ssh "$SSH_TARGET" "
        pkill -SIGTERM -f llama-server 2>/dev/null || true
        sleep 2
        pkill -SIGKILL -f llama-server 2>/dev/null || true
        sleep 1
        for i in \$(seq 1 5); do
            ss -tlnp 2>/dev/null | grep -q ':${SERVER_PORT}' || break
            sleep 2
        done
    " 2>/dev/null || true
}

# clean_edge: aggressive teardown including any companion containers.
clean_edge() {
    ssh "$SSH_TARGET" "
        pkill -f llama-server 2>/dev/null || true
        cd ~/projects/raspi-claw 2>/dev/null && docker compose down 2>/dev/null || true
    " 2>/dev/null || true
    sleep 4
}

# ── Start ─────────────────────────────────────────────────────────────

# start_server: launch llama-server with a given context size.
# Sets CURRENT_CTX_SIZE; MODEL_FILE must be set by caller.
start_server() {
    local ctx="${1:-${CURRENT_CTX_SIZE}}"
    local n_predict="${2:-${N_PREDICT_MATH}}"
    CURRENT_CTX_SIZE="$ctx"

    log_ts "  Starting llama-server: model=${MODEL_FILE} ctx=${ctx} threads=${THREADS} n_predict=${n_predict}"
    track_step "start_server_ctx${ctx}"

    ssh "$SSH_TARGET" "bash -c '
        cd ${REMOTE_DIR} || exit 1
        nohup ./llama-server \
            -m model/${MODEL_FILE} \
            --host 0.0.0.0 \
            --port ${SERVER_PORT} \
            --ctx-size ${ctx} \
            --threads ${THREADS} \
            -n ${n_predict} \
            > llama-server.log 2>&1 &
        disown \$! 2>/dev/null || true
        echo started
    '" 2>/dev/null || true
}

# ── Restart with new context size ────────────────────────────────────
# Used by the escalation loop to move from one ctx level to the next.
# Flushes OS page cache after stop to reclaim model weight pages.
restart_server_with_ctx() {
    local ctx="$1"
    local ram_before; ram_before=$(measure_ram_mb)
    log_ts "  [ctx-restart] Stopping server (RAM=${ram_before} MB) → ctx=${ctx}..."
    track_step "restart_ctx${ctx}"

    ssh "$SSH_TARGET" "
        pkill -SIGTERM -f llama-server 2>/dev/null || true
        sleep 3
        pkill -SIGKILL -f llama-server 2>/dev/null || true
        sleep 2
        sync 2>/dev/null || true
        sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || \
            sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
        sleep 1
    " 2>/dev/null || true

    start_server "$ctx"

    if ! wait_for_server; then
        log_ts "  [ctx-restart] !! Server failed to start with ctx=${ctx}"
        return 1
    fi

    local ram_after; ram_after=$(measure_ram_mb)
    log_ts "  [ctx-restart] Server ready ctx=${ctx} RAM: ${ram_before} → ${ram_after} MB"
    return 0
}

# ── Model transfer ────────────────────────────────────────────────────

transfer_model() {
    local hf_repo="$1"
    local model_file="$2"
    local model_path="${LOCAL_MODEL_DIR}/${model_file}"

    ssh "$SSH_TARGET" "mkdir -p ${REMOTE_MODEL_DIR}" 2>/dev/null || true

    if ssh "$SSH_TARGET" "[ -f ${REMOTE_MODEL_DIR}/${model_file} ]" 2>/dev/null; then
        log "  Model already present on edge device."
        return 0
    fi

    if [ ! -f "$model_path" ]; then
        log "  Downloading from HuggingFace: ${hf_repo} / ${model_file} ..."
        mkdir -p "$LOCAL_MODEL_DIR"
        if ! huggingface-cli download "$hf_repo" "$model_file" --local-dir "$LOCAL_MODEL_DIR"; then
            log "  ERROR: Download failed — skipping model."
            return 1
        fi
    fi

    log "  Transferring to edge device (this may take a few minutes)..."
    track_step "scp_transfer"
    scp "$model_path" "${SSH_TARGET}:${REMOTE_MODEL_DIR}/"
    log "  Transfer complete."
    return 0
}

# ── KV cache inspection ───────────────────────────────────────────────

get_kv_cache_mb() {
    ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no \
        "$SSH_TARGET" \
        "grep -i 'kv self size\|kv buffer\|KV buffer' \
         ${REMOTE_DIR}/llama-server.log 2>/dev/null | \
         grep -oP '[\d.]+(?=\s*MiB)' | head -1" 2>/dev/null || echo "N/A"
}
