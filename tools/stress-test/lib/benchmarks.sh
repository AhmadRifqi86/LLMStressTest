#!/usr/bin/env bash
# ── benchmarks.sh ─────────────────────────────────────────────────────
# Exponential benchmark runners for all seven evaluation sections.
#
# ESCALATION MODEL
# ─────────────────
# For a dataset of N questions and EXPONENTIAL_ROUNDS doubling rounds:
#   base_q = ceil(N / 2^(R-1))
#   round k (1-based): n_q = min(N, base_q × 2^(k-1))
#
# Example — N=100, R=6:  4 → 8 → 16 → 32 → 64 → 100
#
# After all rounds complete at one context level, the server is
# restarted with the next CTX_ESCALATION size and rounds repeat.
# This drives exponential growth in both question count and KV cache,
# ensuring OOM is reached even on models that survive small contexts.
#
# RESULT GLOBALS (set by each run_*_exponential function)
# ────────────────────────────────────────────────────────
#   BENCH_OOM_ROUND   — global round number that triggered OOM (0 = none)
#   BENCH_OOM_CTX     — context size at OOM
#   BENCH_OOM_N_Q     — question/para count at OOM
#   BENCH_TOTAL_ROUNDS — total rounds completed (including OOM round)
#   BENCH_CORRECT     — cumulative correct answers across all rounds
#   BENCH_TOTAL_Q     — cumulative questions answered across all rounds
#
# INTERNAL ROUND GLOBALS (set before each API call for per-question log)
# ────────────────────────────────────────────────────────────────────────
#   Q_ROUND_CORRECT   — correct answers in the current question loop
#   Q_ROUND_TOTAL     — questions run in the current question loop

BENCH_OOM_ROUND=0
BENCH_OOM_CTX=0
BENCH_OOM_N_Q=0
BENCH_TOTAL_ROUNDS=0
BENCH_CORRECT=0
BENCH_TOTAL_Q=0

Q_ROUND_CORRECT=0
Q_ROUND_TOTAL=0

# ── Helper: compute exponential round sizes from dataset length ───────
# Prints one integer per line: base, base*2, base*4, ... capped at total.
_compute_round_sizes() {
    local total="$1"
    local rounds="$2"
    python3 - "$total" "$rounds" << 'PYEOF'
import sys, math
total  = int(sys.argv[1])
rounds = int(sys.argv[2])
base   = max(1, math.ceil(total / (2 ** (rounds - 1))))
for r in range(rounds):
    print(min(total, base * (2 ** r)))
PYEOF
}

# ── Inner loop: multiple-choice questions ────────────────────────────
# Writes results to Q_ROUND_CORRECT / Q_ROUND_TOTAL (no subshell needed).
# Args: n_q  sys_prompt  n_predict  qs_var_name  ans_var_name
_run_mc_questions() {
    local n_q="$1"
    local sys_prompt="$2"
    local n_predict="$3"
    local qs_var="$4"
    local ans_var="$5"
    local -n _qs="$qs_var"
    local -n _ans="$ans_var"

    Q_ROUND_CORRECT=0
    Q_ROUND_TOTAL=0

    local total_in_array="${#_qs[@]}"
    local limit=$(( n_q < total_in_array ? n_q : total_in_array ))

    for (( i = 0; i < limit; i++ )); do
        local qnum=$(( i + 1 ))
        CURRENT_Q="Q${qnum}"
        track_step "api_call"

        local q_ts; q_ts=$(iso_ts)
        local snap_pre; snap_pre=$(snapshot_resources)
        local cpu_pre; cpu_pre=$(echo "$snap_pre" | awk '{print $1}')
        local ram_pre; ram_pre=$(echo "$snap_pre" | awk '{print $2}')

        call_api "$sys_prompt" "$(printf '%b' "${_qs[$i]}")" "$n_predict"

        local snap_post; snap_post=$(snapshot_resources)
        local cpu_post; cpu_post=$(echo "$snap_post" | awk '{print $1}')
        local ram_post; ram_post=$(echo "$snap_post" | awk '{print $2}')

        local predicted; predicted=$(parse_mc "$LAST_RESPONSE")
        local correct_ans="${_ans[$i]}"
        local ok=0; [ "$predicted" = "$correct_ans" ] && ok=1
        Q_ROUND_CORRECT=$(( Q_ROUND_CORRECT + ok ))
        Q_ROUND_TOTAL=$(( Q_ROUND_TOTAL + 1 ))

        local marker; marker=$([ "$ok" -eq 1 ] && echo "✓" || echo "✗")
        log "    Q${qnum} [${q_ts}] [ctx=${CURRENT_CTX_SIZE} round=${CURRENT_ROUND}]: predicted=${predicted} correct=${correct_ans} ${marker}  latency=${LAST_LATENCY_MS}ms  cpu=${cpu_pre}→${cpu_post}%  ram=${ram_pre}→${ram_post}MB  prompt_tok=${LAST_PROMPT_TOKENS:-?}  compl_tok=${LAST_COMPLETION_TOKENS:-?}"
        log "           Response: $(echo "$LAST_RESPONSE" | head -1 | cut -c1-100)"
    done
}

# ── Inner loop: GSM8K (uses parse_number instead of parse_mc) ─────────
# Args: n_q  qs_var_name  ans_var_name
_run_gsm_questions() {
    local n_q="$1"
    local qs_var="$2"
    local ans_var="$3"
    local -n _qs="$qs_var"
    local -n _ans="$ans_var"

    Q_ROUND_CORRECT=0
    Q_ROUND_TOTAL=0

    local total_in_array="${#_qs[@]}"
    local limit=$(( n_q < total_in_array ? n_q : total_in_array ))

    for (( i = 0; i < limit; i++ )); do
        local qnum=$(( i + 1 ))
        CURRENT_Q="GSM_Q${qnum}"
        track_step "api_call"

        local q_ts; q_ts=$(iso_ts)
        local snap_pre; snap_pre=$(snapshot_resources)
        local cpu_pre; cpu_pre=$(echo "$snap_pre" | awk '{print $1}')
        local ram_pre; ram_pre=$(echo "$snap_pre" | awk '{print $2}')

        call_api "$GSM8K_SYSTEM" "${_qs[$i]}" "$N_PREDICT_MATH"

        local snap_post; snap_post=$(snapshot_resources)
        local cpu_post; cpu_post=$(echo "$snap_post" | awk '{print $1}')
        local ram_post; ram_post=$(echo "$snap_post" | awk '{print $2}')

        local predicted; predicted=$(parse_number "$LAST_RESPONSE")
        local correct_ans="${_ans[$i]}"
        local ok=0; [ "$predicted" = "$correct_ans" ] && ok=1
        Q_ROUND_CORRECT=$(( Q_ROUND_CORRECT + ok ))
        Q_ROUND_TOTAL=$(( Q_ROUND_TOTAL + 1 ))

        local marker; marker=$([ "$ok" -eq 1 ] && echo "✓" || echo "✗")
        log "    Q${qnum} [${q_ts}] [ctx=${CURRENT_CTX_SIZE} round=${CURRENT_ROUND}]: predicted=${predicted} correct=${correct_ans} ${marker}  latency=${LAST_LATENCY_MS}ms  cpu=${cpu_pre}→${cpu_post}%  ram=${ram_pre}→${ram_post}MB"
        log "           Response: $(echo "$LAST_RESPONSE" | tail -3 | tr '\n' ' ' | cut -c1-120)"
    done
}

# ── Exponential benchmark runner (MC + GSM8K) ─────────────────────────
# Args: bench_name  qs_var  ans_var  sys_prompt  bench_type  n_predict
#   bench_type: "mc" for multiple-choice, "gsm" for GSM8K math
# Returns 1 if OOM was triggered, 0 if all rounds completed cleanly.
run_benchmark_exponential() {
    local bench_name="$1"
    local qs_var="$2"
    local ans_var="$3"
    local sys_prompt="$4"
    local bench_type="$5"
    local n_predict="$6"

    BENCH_OOM_ROUND=0; BENCH_OOM_CTX=0; BENCH_OOM_N_Q=0
    BENCH_TOTAL_ROUNDS=0; BENCH_CORRECT=0; BENCH_TOTAL_Q=0

    # Resolve dataset size without a nameref (compatible with bash 4.3+)
    local total_q
    total_q=$(eval "echo \${#${qs_var}[@]}")

    sep
    log "  [BENCH] ${bench_name} — exponential OOM stress"
    log "         Dataset  : ${total_q} questions  (source: qcache.sh)"
    log "         Rounds   : ${EXPONENTIAL_ROUNDS} doubling rounds per ctx level"
    log "         Ctx levels: ${CTX_ESCALATION[*]}"
    log "         Goal     : trigger OOM — record exact round / ctx / n_q"
    sep

    # Compute question counts for each round based on actual dataset size
    mapfile -t _round_sizes < <(_compute_round_sizes "$total_q" "$EXPONENTIAL_ROUNDS")

    local global_round=0

    for ctx in "${CTX_ESCALATION[@]}"; do
        CURRENT_CTX_SIZE="$ctx"

        if ! restart_server_with_ctx "$ctx"; then
            log "  !! Server failed to start with ctx=${ctx} — skipping this ctx level"
            continue
        fi

        local kv_mb; kv_mb=$(get_kv_cache_mb)
        log_ts "  ctx=${ctx}: KV cache pre-allocated ${kv_mb} MiB"

        mark_section "${bench_name}"

        for n_q in "${_round_sizes[@]}"; do
            global_round=$(( global_round + 1 ))
            CURRENT_ROUND="$global_round"
            BENCH_TOTAL_ROUNDS="$global_round"

            local ram_before; ram_before=$(measure_ram_mb)
            local round_start_ts; round_start_ts=$(iso_ts)
            local round_t0; round_t0=$(date +%s)

            log ""
            log "  ┌── Round ${global_round} ──────────────────────────────────────────────────"
            log "  │  Benchmark  : ${bench_name}"
            log "  │  Started    : ${round_start_ts}"
            log "  │  Ctx size   : ${ctx} tokens  (KV cache: ${kv_mb} MiB)"
            log "  │  Questions  : ${n_q} / ${total_q}  ($(python3 -c "print(f'{${n_q}/${total_q}*100:.0f}')") %  of dataset)"
            log "  │  RAM before : ${ram_before} MB"
            log "  └──────────────────────────────────────────────────────────────────"

            if [ "$bench_type" = "gsm" ]; then
                _run_gsm_questions "$n_q" "$qs_var" "$ans_var"
            else
                _run_mc_questions "$n_q" "$sys_prompt" "$n_predict" "$qs_var" "$ans_var"
            fi

            BENCH_CORRECT=$(( BENCH_CORRECT + Q_ROUND_CORRECT ))
            BENCH_TOTAL_Q=$(( BENCH_TOTAL_Q + Q_ROUND_TOTAL ))

            local round_t1; round_t1=$(date +%s)
            local round_elapsed=$(( round_t1 - round_t0 ))
            local ram_after; ram_after=$(measure_ram_mb)
            local ram_delta=$(( ram_after - ram_before ))
            local pct="N/A"
            [ "${Q_ROUND_TOTAL:-0}" -gt 0 ] && \
                pct=$(python3 -c "print(f'{${Q_ROUND_CORRECT}/${Q_ROUND_TOTAL}*100:.0f}')")

            log ""
            log "  Round ${global_round} complete [$(iso_ts)]:"
            log "    Result   : ${Q_ROUND_CORRECT}/${Q_ROUND_TOTAL} = ${pct}%"
            log "    Duration : $(fmt_time "$round_elapsed")"
            log "    RAM      : ${ram_before} → ${ram_after} MB  (Δ${ram_delta} MB)"

            # OOM check — if SSH is gone, record the threshold and return
            if ! assert_alive; then
                BENCH_OOM_ROUND="$global_round"
                BENCH_OOM_CTX="$ctx"
                BENCH_OOM_N_Q="$n_q"
                log ""
                log "  !! OOM THRESHOLD REACHED"
                log "     Benchmark : ${bench_name}"
                log "     Round     : ${global_round}"
                log "     Ctx size  : ${ctx} tokens"
                log "     Questions : ${n_q}"
                log "     RAM at    : ${ram_after} MB"
                return 1
            fi
        done
    done

    log ""
    log "  ✓ ${bench_name}: completed all ${global_round} rounds without OOM"
    log "    Cumulative — answered: ${BENCH_TOTAL_Q}  correct: ${BENCH_CORRECT}"
    return 0
}

# ══════════════════════════════════════════════════════════════════════
#  Long Context / Needle-in-a-Haystack exponential runner
#  Kamradt (2023) methodology; Liu et al. (2023) arXiv:2307.03172
#
#  Escalation: each ctx level adds more haystack paragraphs (~100 tokens
#  each), so both KV cache AND prompt length grow together.
#  At each level, three needle positions are tested: begin / middle / end.
# ══════════════════════════════════════════════════════════════════════

NEEDLE_FACT="[SYSTEM NOTE] The primary authentication code for Project Helios is XK-7749."
LC_QUESTION="Based only on the information provided above, what is the authentication code for Project Helios? Reply with only the code, nothing else."

# Haystack paragraphs — edge-AI domain, ~100 tokens each
_HP=(
"Edge computing distributes computational workloads closer to the data source, reducing latency and bandwidth consumption in IoT deployments. Rather than routing all sensor data to centralized cloud servers, edge nodes perform local inference and transmit only summarized results upstream. This approach benefits time-sensitive applications such as industrial control, autonomous navigation, and real-time health monitoring. Small language models running on embedded hardware enable natural language interfaces without cloud dependency."
"The Raspberry Pi 5 features a quad-core ARM Cortex-A76 processor running at 2.4 GHz, offering substantially improved integer performance over its predecessors. With up to 8 GB of LPDDR4X memory and a PCIe 2.0 interface, the platform supports modest inference workloads using quantized neural network models. Thermal management is critical during sustained inference; the official active cooling solution maintains junction temperatures below 80 degrees Celsius under continuous load. USB 3.0 connectivity enables fast model transfer from host machines during benchmarking."
"Transformer architectures rely on the self-attention mechanism to model token dependencies regardless of their distance in a sequence. The multi-head attention operation computes query, key, and value projections for each token, then takes a weighted sum of values based on query-key dot products. As sequence length grows, attention complexity scales quadratically, making long-context inference particularly expensive. Modern optimizations such as grouped-query attention and sliding-window attention reduce this overhead at the cost of modeling fidelity at extreme distances."
"Post-training quantization reduces model weight precision from 16-bit floating point to lower bitwidths such as 4-bit or 2-bit integers. The GGUF format supports schemes including Q4_K_M, Q5_K_M, and Q8_0, each trading accuracy for reduced memory footprint and faster integer arithmetic. On CPUs without dedicated matrix acceleration, quantized kernels achieve throughput gains by fitting more weights in L2 and L3 cache. Calibration data quality significantly influences the accuracy retained after aggressive quantization to low bitwidths."
"The MQTT protocol is widely adopted for lightweight publish-subscribe messaging in IoT deployments. Operating over TCP with a minimal two-byte fixed header, MQTT supports three quality-of-service levels: at-most-once, at-least-once, and exactly-once delivery semantics. Retained messages allow newly connected subscribers to immediately receive the last published value for a topic. Combined with TLS encryption and client certificate authentication, MQTT provides a secure and efficient transport layer for telemetry and command channels in distributed sensor networks."
"The key-value cache in transformer inference stores previously computed attention keys and values for each layer, enabling efficient autoregressive generation without recomputing representations for prior tokens. Cache size scales with context length, number of layers, number of KV heads, and the head dimension. A 2048-token context with a 32-layer model using 8 KV heads of dimension 128 requires approximately 128 MB of contiguous memory. Pre-allocating this cache at server startup ensures deterministic memory usage and avoids fragmentation during extended sessions."
"GGUF is a binary container format designed for efficient storage and loading of quantized language model weights. It supersedes the earlier GGML format by embedding all necessary metadata—tokenizer vocabulary, model hyperparameters, and quantization descriptors—within a single self-contained file. The format supports memory-mapped loading, allowing the operating system to page in only the weight tensors required for the current computation. This property is especially valuable on memory-constrained edge devices where loading an entire model into RAM would otherwise be prohibitive."
"Federated learning trains a global model by aggregating gradient updates from multiple edge clients without centralizing raw data. Each participating device computes local gradients on its private dataset and transmits only the compressed update vector to a coordinating server. Differential privacy mechanisms add calibrated noise to individual updates, providing formal privacy guarantees. Communication efficiency techniques such as gradient sparsification and quantized communication reduce the bandwidth overhead inherent in large model updates across heterogeneous edge networks with limited uplink capacity."
"Memory bandwidth is frequently the binding constraint for language model inference on consumer and embedded hardware. A 4-bit quantized 3-billion-parameter model requires approximately 1.5 GB of weight data; at 25 GB per second memory bandwidth the theoretical minimum time to read all weights once is about 60 milliseconds per token. ARM Cortex-A76 cores share a unified L3 cache of approximately 2 MB, insufficient to hold more than a small fraction of the weight matrices for even small models. Prefetching strategies and tiled matrix-vector multiplication layouts partially mitigate resulting cache misses."
"Quantization-aware training incorporates the effect of quantization into the training loss, producing models that are intrinsically more robust to low-precision representation. Unlike post-training quantization, which applies discretization after training is complete, quantization-aware training allows the model to adapt its weight distributions to minimize accuracy loss under quantization constraints. The straight-through estimator allows gradients to flow through the otherwise non-differentiable rounding operation, enabling end-to-end optimization of quantized networks."
"Sparse attention mechanisms selectively attend to subsets of tokens rather than all pairs, reducing the quadratic complexity of full attention to sub-quadratic in sequence length. Patterns including local windowed attention, strided global tokens, and learned sparse patterns have been explored extensively in the literature. BigBird and Longformer introduced sparse attention frameworks suitable for sequences up to 16,000 tokens on commodity hardware. These techniques are particularly relevant for retrieval-augmented generation pipelines that concatenate long document contexts with user queries."
"On-device fine-tuning enables personalization of language models to individual users without transmitting raw data to cloud servers. Techniques such as adapter modules and low-rank adaptation insert small trainable matrices into frozen model layers, reducing the number of trainable parameters to a small fraction of the full model. Combined with gradient checkpointing and mixed-precision arithmetic, on-device fine-tuning has been demonstrated on devices with 4 GB of RAM. Privacy guarantees are maintained by keeping training data and gradient updates local to the device."
)

# _build_lc_prompt: compose a prompt with needle at a given position.
# Args: n_paras  position("begin"|"middle"|"end")
# Outputs the full prompt text to stdout.
_build_lc_prompt() {
    local n_paras="$1"
    local position="$2"
    local max_paras="${#_HP[@]}"

    local n_before n_after
    case "$position" in
        begin)  n_before=0;                          n_after=$(( n_paras ))          ;;
        middle) n_before=$(( n_paras / 2 ));         n_after=$(( n_paras - n_before )) ;;
        end)    n_before=$(( n_paras ));              n_after=0                       ;;
        *)      n_before=$(( n_paras / 2 ));         n_after=$(( n_paras - n_before )) ;;
    esac

    local prompt="" idx=0

    for (( p = 0; p < n_before && idx < max_paras; p++, idx++ )); do
        prompt+="${_HP[$idx]}"$'\n\n'
    done
    prompt+="${NEEDLE_FACT}"$'\n\n'
    for (( p = 0; p < n_after && idx < max_paras; p++, idx++ )); do
        prompt+="${_HP[$idx]}"$'\n\n'
    done
    prompt+="${LC_QUESTION}"

    printf '%s' "$prompt"
}

# run_long_context_exponential: escalate haystack size and ctx together.
# Each CTX_ESCALATION level adds more paragraphs around the needle.
# Three positions are tested per level: begin / middle / end.
# Returns 1 on OOM, 0 if all levels complete cleanly.
run_long_context_exponential() {
    BENCH_OOM_ROUND=0; BENCH_OOM_CTX=0; BENCH_OOM_N_Q=0
    BENCH_TOTAL_ROUNDS=0; BENCH_CORRECT=0; BENCH_TOTAL_Q=0

    local max_paras="${#_HP[@]}"
    # Para counts grow with ctx level index (exponential-ish, capped at max)
    # Level 0: 1 para, Level 1: 2, Level 2: 4, Level 3: 8, Level 4: max
    local -a para_counts
    local lvl_count="${#CTX_ESCALATION[@]}"
    for (( lvl = 0; lvl < lvl_count; lvl++ )); do
        local pc=$(( 1 << lvl ))   # 1, 2, 4, 8, 16, ...
        [ "$pc" -gt "$max_paras" ] && pc="$max_paras"
        para_counts+=("$pc")
    done

    local positions=("begin" "middle" "end")
    local global_round=0

    sep
    log "  [BENCH] LongContext — exponential OOM stress  [Kamradt 2023; Liu et al. 2023]"
    log "         Needle    : '${NEEDLE_FACT:0:60}...'"
    log "         Positions : begin / middle / end"
    log "         Ctx levels: ${CTX_ESCALATION[*]}"
    log "         Para counts per level: ${para_counts[*]}  (max: ${max_paras})"
    sep

    for (( lvl = 0; lvl < lvl_count; lvl++ )); do
        local ctx="${CTX_ESCALATION[$lvl]}"
        local n_paras="${para_counts[$lvl]}"
        CURRENT_CTX_SIZE="$ctx"

        if ! restart_server_with_ctx "$ctx"; then
            log "  !! LongCtx: server failed at ctx=${ctx} — skipping level"
            continue
        fi

        local kv_mb; kv_mb=$(get_kv_cache_mb)
        log_ts "  ctx=${ctx}: KV cache ${kv_mb} MiB  haystack=${n_paras} paras (~$(( n_paras * 100 )) tokens)"

        mark_section "LongCtx"

        for pos in "${positions[@]}"; do
            global_round=$(( global_round + 1 ))
            CURRENT_ROUND="$global_round"
            BENCH_TOTAL_ROUNDS="$global_round"
            CURRENT_Q="lc_${pos}"
            track_step "api_call"

            local prompt; prompt=$(_build_lc_prompt "$n_paras" "$pos")
            local q_ts; q_ts=$(iso_ts)
            local snap_pre; snap_pre=$(snapshot_resources)
            local cpu_pre; cpu_pre=$(echo "$snap_pre" | awk '{print $1}')
            local ram_before; ram_before=$(echo "$snap_pre" | awk '{print $2}')

            log ""
            log "  ┌── LongCtx Round ${global_round} ─────────────────────────────────────────"
            log "  │  Started    : ${q_ts}"
            log "  │  Ctx size   : ${ctx} tokens  (KV: ${kv_mb} MiB)"
            log "  │  Paragraphs : ${n_paras}  (~$(( n_paras * 100 )) haystack tokens)"
            log "  │  Position   : ${pos}"
            log "  │  RAM before : ${ram_before} MB"
            log "  └──────────────────────────────────────────────────────────────────"

            call_api "$LC_SYSTEM" "$prompt" "$N_PREDICT_LC"

            local snap_post; snap_post=$(snapshot_resources)
            local cpu_post; cpu_post=$(echo "$snap_post" | awk '{print $1}')
            local ram_after; ram_after=$(echo "$snap_post" | awk '{print $2}')
            local ram_delta=$(( ram_after - ram_before ))

            local found=0
            echo "$LAST_RESPONSE" | grep -qF "XK-7749" && found=1
            BENCH_CORRECT=$(( BENCH_CORRECT + found ))
            BENCH_TOTAL_Q=$(( BENCH_TOTAL_Q + 1 ))

            local marker; marker=$([ "$found" -eq 1 ] && echo "FOUND" || echo "MISS")
            local ctx_fill="?"
            [ "${LAST_PROMPT_TOKENS:-0}" -gt 0 ] && \
                ctx_fill=$(python3 -c "print(f'{${LAST_PROMPT_TOKENS}/${ctx}*100:.0f}')")

            log "  needle=${marker}  latency=${LAST_LATENCY_MS}ms  ctx_fill=${ctx_fill}%  cpu=${cpu_pre}→${cpu_post}%  ram=${ram_before}→${ram_after}MB (Δ${ram_delta}MB)  prompt_tok=${LAST_PROMPT_TOKENS:-?}  compl_tok=${LAST_COMPLETION_TOKENS:-?}"
            log "  Response: $(echo "$LAST_RESPONSE" | head -1 | cut -c1-100)"

            if ! assert_alive; then
                BENCH_OOM_ROUND="$global_round"
                BENCH_OOM_CTX="$ctx"
                BENCH_OOM_N_Q="$n_paras"
                log ""
                log "  !! OOM THRESHOLD REACHED"
                log "     Benchmark : LongCtx"
                log "     Round     : ${global_round}"
                log "     Ctx size  : ${ctx} tokens"
                log "     Paragraphs: ${n_paras}  position: ${pos}"
                log "     RAM at    : ${ram_after} MB"
                return 1
            fi
        done
    done

    log ""
    log "  ✓ LongCtx: completed all ${global_round} rounds without OOM"
    log "    Needle found ${BENCH_CORRECT} / ${BENCH_TOTAL_Q} times across all rounds"
    return 0
}
