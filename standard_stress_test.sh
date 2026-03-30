#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
#  standard_stress_test.sh — SLM evaluation using established benchmarks
#
#  REFERENCES (evaluation methodology and question sources):
#
#  [1] ARC — AI2 Reasoning Challenge (0-shot, MC)
#      Clark, P., et al. (2018). "Think you have solved question
#      answering? Try ARC, the AI2 reasoning challenge."
#      arXiv:1803.05457
#
#  [2] HellaSwag — sentence completion (0-shot, MC)
#      Zellers, R., et al. (2019). "HellaSwag: Can a machine really
#      finish your sentence?" arXiv:1905.07830
#
#  [3] MMLU — Massive Multitask Language Understanding (5-shot, MC)
#      Hendrycks, D., et al. (2020). "Measuring massive multitask
#      language understanding." arXiv:2009.03300
#
#  [4] GSM8K — Grade School Math (2-shot chain-of-thought)
#      Cobbe, K., et al. (2021). "Training verifiers to solve math
#      word problems." arXiv:2110.14168
#
#  [5] TruthfulQA — factuality (0-shot, MC1 format)
#      Lin, S., et al. (2021). "TruthfulQA: Measuring how models
#      mimic human falsehoods." arXiv:2109.07958
#
#  [6] Open LLM Leaderboard (published baseline scores used here)
#      Beeching, E., et al. (2023). HuggingFace.
#
#  [7] Long Context / Needle-in-a-Haystack
#      Kamradt, G. (2023). "Needle in a Haystack — Pressure Testing LLMs."
#      https://github.com/gkamradt/LLMTest_NeedleInAHaystack
#      Liu, N.F., et al. (2023). "Lost in the Middle: How Language Models
#      Use Long Contexts." arXiv:2307.03172
#      Three tests: needle at beginning (LC1 ~400 tok), middle (LC2 ~900 tok),
#      end (LC3 ~1200 tok). Eval: response contains exact code "XK-7749".
#      Also reports: KV cache pre-allocation (MB), prompt token fill (%),
#      per-prompt RAM delta (MB).
#
#  EVALUATION PROTOCOL (matching standard practice):
#    - ARC-Easy / ARC-Challenge : 0-shot, normalized accuracy
#    - HellaSwag                : 0-shot, normalized accuracy
#    - MMLU                     : 5-shot, normalized accuracy
#    - GSM8K                    : 2-shot chain-of-thought, exact match
#    - TruthfulQA               : 0-shot, MC1 accuracy
#    - Long Context (Needle)    : needle retrieval at 3 context positions
#
#  COMPOSITE SCORE:
#    Mean accuracy across all 7 sections (0–100 %)
#    Reported alongside published full-dataset baselines for comparison.
#
#  USAGE:
#    ./standard_stress_test.sh                   — auto top-3 from newest report
#    ./standard_stress_test.sh llama3.2-1b       — specific model(s)
# ═══════════════════════════════════════════════════════════════════════
set -e

# ── SSH / server config ───────────────────────────────────────────────
SSH_USER="mitlab"
SSH_HOST="192.168.50.233"
SSH_TARGET="${SSH_USER}@${SSH_HOST}"
REMOTE_DIR="~/projects/llamacpp"
REMOTE_MODEL_DIR="${REMOTE_DIR}/model"
LOCAL_MODEL_DIR="./llm"

SERVER_PORT=8080
CTX_SIZE=2048
THREADS=4
N_PREDICT_MC=64        # Multiple-choice: only needs a single letter
N_PREDICT_MATH=512     # GSM8K CoT needs room for step-by-step reasoning
N_PREDICT_LC=64        # Long-context needle: short answer expected
SERVER_READY_TIMEOUT=120

# ── OOM-resistance: proactive server restart every N questions ─────────
# Set to 0 to disable proactive restarts (reactive restart on ERROR still active).
# 10 is a safe default for a 2 GB Pi; increase if your Pi has more RAM.
BATCH_RESTART=10

# ── RAM threshold: restart early if used RAM exceeds this before a question ──
# Based on: llama3.2-1b at idle after load ≈ 1460 MB on Pi 4 2GB.
# Trigger a restart before the question if used RAM is already this high.
# Set to 0 to disable threshold-based restarts.
RAM_RESTART_THRESHOLD_MB=1450

# ── Report files ──────────────────────────────────────────────────────
TS=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="standard_stress_${TS}.txt"
STEP_FILE="standard_stress_laststep.txt"
HEARTBEAT_PID=""
CURRENT_MODEL="none"
CURRENT_SECTION="none"
CURRENT_Q="none"

# ── Resource sampler state ────────────────────────────────────────────
RESOURCE_SAMPLER_PID=""
RESOURCE_SAMPLE_FILE=""
# Per-model resource results (set by stop_resource_sampler)
CPU_AVG="N/A"; CPU_PEAK="N/A"
RAM_AVG="N/A"; RAM_PEAK="N/A"

log()  { echo "$@"; echo "$@" >> "$REPORT_FILE"; }
sep()  { log "────────────────────────────────────────────────────────────────────"; }
sep2() { log "════════════════════════════════════════════════════════════════════"; }

# ─────────────────────────────────────────────────────────────────────
#  Model list  (HF_REPO|GGUF_FILE|CONFIG_NAME)
# ─────────────────────────────────────────────────────────────────────
ALL_MODELS=(
    "Qwen/Qwen2.5-0.5B-Instruct-GGUF|qwen2.5-0.5b-instruct-q4_k_m.gguf|qwen2.5-0.5b"
    "bartowski/Qwen_Qwen3-0.6B-GGUF|Qwen_Qwen3-0.6B-Q4_K_M.gguf|qwen3-0.6b"
    "bartowski/Qwen_Qwen3.5-0.8B-GGUF|Qwen_Qwen3.5-0.8B-Q8_0.gguf|qwen3.5-0.8b"
    "bartowski/Qwen_Qwen3-1.7B-GGUF|Qwen_Qwen3-1.7B-Q4_K_M.gguf|qwen3-1.7b"
    "bartowski/Qwen_Qwen3-4B-GGUF|Qwen_Qwen3-4B-Q4_K_M.gguf|qwen3-4b"
    "bartowski/Llama-3.2-1B-Instruct-GGUF|Llama-3.2-1B-Instruct-Q4_K_M.gguf|llama3.2-1b"
    "bartowski/Llama-3.2-3B-Instruct-GGUF|Llama-3.2-3B-Instruct-Q4_K_M.gguf|llama3.2-3b"
    "bartowski/SmolLM2-1.7B-Instruct-GGUF|SmolLM2-1.7B-Instruct-Q4_K_M.gguf|smollm2-1.7b"
    "bartowski/gemma-2-2b-it-GGUF|gemma-2-2b-it-Q4_K_M.gguf|gemma2-2b"
    "bartowski/google_gemma-3-1b-it-GGUF|google_gemma-3-1b-it-Q4_K_M.gguf|gemma3-1b"
)

# Published baselines from Open LLM Leaderboard [6] (full-dataset, %)
# Format: ARC_E|ARC_C|HellaSwag|MMLU|GSM8K|TruthfulQA
declare -A BASELINES
BASELINES["llama3.2-1b"]="71|46|61|46|44|45"
BASELINES["llama3.2-3b"]="78|54|74|63|74|52"
BASELINES["gemma2-2b"]="80|60|73|52|30|71"
BASELINES["gemma3-1b"]="76|52|70|54|36|66"
BASELINES["smollm2-1.7b"]="66|45|66|50|20|48"
BASELINES["qwen3-0.6b"]="60|38|55|44|30|42"
BASELINES["qwen3-1.7b"]="72|50|68|55|45|50"
BASELINES["qwen3-4b"]="82|65|80|72|80|58"
BASELINES["qwen2.5-0.5b"]="58|35|52|42|25|40"
BASELINES["qwen3.5-0.8b"]="63|40|58|46|32|44"

# ─────────────────────────────────────────────────────────────────────
#  Model selection — top 3 from newest benchmark report, or manual
# ─────────────────────────────────────────────────────────────────────
if [ "$#" -gt 0 ]; then
    MODELS=()
    for entry in "${ALL_MODELS[@]}"; do
        cfg=$(echo "$entry" | cut -d'|' -f3)
        for arg in "$@"; do
            [ "$cfg" = "$arg" ] && MODELS+=("$entry")
        done
    done
    if [ "${#MODELS[@]}" -eq 0 ]; then
        echo "ERROR: No match for: $*"
        echo "Available: $(for e in "${ALL_MODELS[@]}"; do echo -n "$(echo "$e"|cut -d'|' -f3) "; done)"
        exit 1
    fi
    log "Manual selection: ${#MODELS[@]} model(s)"
else
    NEWEST_REPORT=$(ls -t report_*.txt 2>/dev/null | head -1 || true)
    if [ -z "$NEWEST_REPORT" ]; then
        echo "ERROR: No benchmark report found (report_*.txt)."
        echo "Run benchmark.sh first, or specify models manually."
        exit 1
    fi
    log "Selecting top-3 from: ${NEWEST_REPORT}"
    TOP3=$(python3 - "$NEWEST_REPORT" << 'PYEOF'
import sys, re
report = open(sys.argv[1]).read()
scores = {}
in_table = False
for line in report.splitlines():
    if 'ToolSel' in line and 'Syntax' in line:
        in_table = True; continue
    if in_table:
        if re.match(r'^[-\s|]+$', line): continue
        if '|' not in line: in_table = False; continue
        cols = [c.strip() for c in line.split('|')]
        if len(cols) < 9: continue
        model = cols[0]
        if not model or model.startswith('-') or model == 'Model': continue
        def pct(s):
            m = re.search(r'([\d.]+)%', s); return float(m.group(1)) if m else 0.0
        def ratio(s):
            m = re.search(r'(\d+)/(\d+)', s)
            return float(m.group(1))/float(m.group(2))*100 if m else 0.0
        def ms(s):
            m = re.search(r'(\d+)ms', s); return int(m.group(1)) if m else 99999
        score = pct(cols[3]) + pct(cols[4]) + ratio(cols[7]) + (100-pct(cols[8])) - ms(cols[5])/1000
        scores[model] = score
ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
for name, _ in ranked[:3]:
    print(name)
PYEOF
)
    if [ -z "$TOP3" ]; then
        echo "ERROR: Could not parse scores from ${NEWEST_REPORT}."
        exit 1
    fi
    MODELS=()
    while IFS= read -r cfg; do
        [ -z "$cfg" ] && continue
        for entry in "${ALL_MODELS[@]}"; do
            if echo "$entry" | cut -d'|' -f3 | grep -qx "$cfg"; then
                MODELS+=("$entry"); log "  + ${cfg}"; break
            fi
        done
    done <<< "$TOP3"
    [ "${#MODELS[@]}" -eq 0 ] && { echo "ERROR: Top-3 models not in ALL_MODELS."; exit 1; }
    log ""
fi

# ═══════════════════════════════════════════════════════════════════════
#  SECTION DEFINITIONS
#  100 real questions per section, downloaded from HuggingFace datasets.
#  Requires: pip install datasets
#  Cache file (standard_stress_qcache.sh) is generated on first run.
# ═══════════════════════════════════════════════════════════════════════

# ── Section 4 & 5 system prompts (few-shot examples, not questions) ───
MMLU_SYSTEM='You are a knowledgeable assistant. Answer each question with only the letter of the correct choice (A, B, C, or D). Here are five examples:

Q: What is the powerhouse of the eukaryotic cell?
(A) Nucleus (B) Ribosome (C) Mitochondria (D) Golgi apparatus
A: C

Q: Which gas makes up the largest percentage of Earth'"'"'s atmosphere?
(A) Oxygen (B) Carbon dioxide (C) Nitrogen (D) Argon
A: C

Q: The Pythagorean theorem states that in a right triangle a²+b²=c², where c is the:
(A) shortest side (B) longest side (hypotenuse) (C) middle side (D) any side
A: B

Q: In which century did the Renaissance begin in Italy?
(A) 13th century (B) 14th century (C) 15th century (D) 16th century
A: B

Q: What is the time complexity of binary search on a sorted array of n elements?
(A) O(n) (B) O(n²) (C) O(log n) (D) O(n log n)
A: C

Now answer the following question with only the letter (A, B, C, or D):'

GSM8K_SYSTEM='Solve each math problem step by step. At the end of your solution, write the final answer on its own line in the format: #### <number>

Example 1:
Problem: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did she sell altogether in April and May?
Solution: Natalia sold 48/2 = 24 clips in May. Altogether she sold 48+24 = 72 clips. #### 72

Example 2:
Problem: Weng earns $12 an hour for babysitting. Yesterday she did 50 minutes of babysitting. How much did she earn?
Solution: Weng earns 12/60 = $0.20 per minute. Working 50 minutes: 0.20 × 50 = $10. #### 10

Now solve the following problem:'

# ── Download 100 real questions per section from HuggingFace datasets ─
# Cached to standard_stress_qcache.sh after first run.
_Q_CACHE="standard_stress_qcache.sh"
if [ ! -f "$_Q_CACHE" ]; then
    echo "Fetching 100 questions/section from HuggingFace datasets (one-time, ~2 min)..."
    echo "  Requires: pip install datasets"
    python3 << 'PYEOF' > "$_Q_CACHE"
import sys, random, re

random.seed(42)
N = 100

try:
    from datasets import load_dataset
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "-q"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    from datasets import load_dataset

def esc(s):
    """Escape for bash double-quoted string; real newlines become literal \\n."""
    s = str(s)
    s = s.replace('\\', '\\\\')
    s = s.replace('\n', '\\n')
    s = s.replace('"', '\\"')
    s = s.replace('$', '\\$')
    s = s.replace('`', '\\`')
    return s

def emit_arr(name, items):
    print(f"{name}=(")
    for x in items:
        print(f'    "{esc(x)}"')
    print(")")

def emit_ans(name, items):
    print(f"{name}=(" + " ".join(f'"{x}"' for x in items) + ")")

lfix = {'1':'A','2':'B','3':'C','4':'D','A':'A','B':'B','C':'C','D':'D'}

# ── ARC-Easy ────────────────────────────────────────────────────────
print("# ARC-Easy — allenai/ai2_arc ARC-Easy test split")
ds = [x for x in load_dataset("allenai/ai2_arc","ARC-Easy",split="test",trust_remote_code=True)
      if len(x['choices']['label']) == 4]
if len(ds) < N:
    raise RuntimeError(f"ARC-Easy: only {len(ds)} 4-choice questions in test split (need {N})")
items = random.sample(ds, N)
qs, ans = [], []
for x in items:
    labs = [lfix.get(l, l) for l in x['choices']['label']]
    opts = '\n'.join(f'({l}) {t}' for l, t in zip(labs, x['choices']['text']))
    qs.append(x['question'] + '\n' + opts)
    ans.append(lfix.get(x['answerKey'], x['answerKey']))
assert len(qs) == N, f"ARC-Easy: got {len(qs)} questions, expected {N}"
emit_arr("ARC_E_Q", qs); emit_ans("ARC_E_ANS", ans)

# ── ARC-Challenge ────────────────────────────────────────────────────
print("\n# ARC-Challenge — allenai/ai2_arc ARC-Challenge test split")
ds = [x for x in load_dataset("allenai/ai2_arc","ARC-Challenge",split="test",trust_remote_code=True)
      if len(x['choices']['label']) == 4]
if len(ds) < N:
    raise RuntimeError(f"ARC-Challenge: only {len(ds)} 4-choice questions in test split (need {N})")
items = random.sample(ds, N)
qs, ans = [], []
for x in items:
    labs = [lfix.get(l, l) for l in x['choices']['label']]
    opts = '\n'.join(f'({l}) {t}' for l, t in zip(labs, x['choices']['text']))
    qs.append(x['question'] + '\n' + opts)
    ans.append(lfix.get(x['answerKey'], x['answerKey']))
assert len(qs) == N, f"ARC-Challenge: got {len(qs)} questions, expected {N}"
emit_arr("ARC_C_Q", qs); emit_ans("ARC_C_ANS", ans)

# ── HellaSwag ────────────────────────────────────────────────────────
print("\n# HellaSwag — Rowan/hellaswag validation split")
ds = list(load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True))
if len(ds) < N:
    raise RuntimeError(f"HellaSwag: only {len(ds)} examples in validation split (need {N})")
items = random.sample(ds, N)
qs, ans = [], []
for x in items:
    opts = '\n'.join(f'({chr(65+i)}) {e.strip()}' for i, e in enumerate(x['endings']))
    qs.append(x['ctx'].strip() + '\n' + opts)
    ans.append(chr(65 + int(x['label'])))
assert len(qs) == N, f"HellaSwag: got {len(qs)} questions, expected {N}"
emit_arr("HS_Q", qs); emit_ans("HS_ANS", ans)

# ── MMLU ────────────────────────────────────────────────────────────
print("\n# MMLU — cais/mmlu all subjects test split")
ds = list(load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True))
if len(ds) < N:
    raise RuntimeError(f"MMLU: only {len(ds)} questions in test split (need {N})")
items = random.sample(ds, N)
qs, ans = [], []
for x in items:
    opts = '\n'.join(f'({chr(65+i)}) {c}' for i, c in enumerate(x['choices']))
    qs.append(x['question'] + '\n' + opts)
    ans.append(chr(65 + x['answer']))
assert len(qs) == N, f"MMLU: got {len(qs)} questions, expected {N}"
emit_arr("MMLU_Q", qs); emit_ans("MMLU_ANS", ans)

# ── GSM8K ────────────────────────────────────────────────────────────
print("\n# GSM8K — openai/gsm8k main test split")
ds = list(load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True))
if len(ds) < N:
    raise RuntimeError(f"GSM8K: only {len(ds)} questions in test split (need {N})")
items = random.sample(ds, N)
qs, ans = [], []
for x in items:
    m = re.search(r'####\s*([\d,]+)', x['answer'])
    qs.append(x['question'])
    ans.append(m.group(1).replace(',', '') if m else '?')
assert len(qs) == N, f"GSM8K: got {len(qs)} questions, expected {N}"
emit_arr("GSM8K_Q", qs); emit_ans("GSM8K_ANS", ans)

# ── TruthfulQA ───────────────────────────────────────────────────────
# Filter: need exactly 1 correct answer + at least 3 wrong answers to form 4-choice MC.
# TruthfulQA validation has 817 questions; virtually all pass the filter (>800 qualify),
# so 100 is always reachable. Raise immediately if the pool is insufficient.
print("\n# TruthfulQA — truthful_qa multiple_choice validation split")
ds = list(load_dataset("truthful_qa", "multiple_choice", split="validation", trust_remote_code=True))
random.shuffle(ds)
pool = []
for x in ds:
    choices = x['mc1_targets']['choices']
    labels  = x['mc1_targets']['labels']   # 1=correct, 0=wrong
    if 1 not in labels:
        continue
    correct_text = choices[labels.index(1)]
    wrong = [c for c, l in zip(choices, labels) if l == 0]
    if len(wrong) < 3:
        continue
    pool.append((x['question'], correct_text, wrong))
if len(pool) < N:
    raise RuntimeError(f"TruthfulQA: only {len(pool)} questions pass 4-choice filter (need {N})")
pool = pool[:N]
qs, ans = [], []
for question, correct_text, wrong in pool:
    four = [correct_text] + random.sample(wrong, 3)
    random.shuffle(four)
    opts = '\n'.join(f'({chr(65+i)}) {c}' for i, c in enumerate(four))
    qs.append(question + '\n' + opts)
    ans.append(chr(65 + four.index(correct_text)))
assert len(qs) == N, f"TruthfulQA: got {len(qs)} questions, expected {N}"
emit_arr("TQA_Q", qs); emit_ans("TQA_ANS", ans)
PYEOF

    if [ ! -s "$_Q_CACHE" ]; then
        echo "ERROR: Question download failed. Install datasets: pip install datasets"
        echo "Then re-run this script."
        exit 1
    fi
    echo "Questions cached → $_Q_CACHE  (delete to refresh)"
fi

# shellcheck disable=SC1090
source "$_Q_CACHE"
echo "Loaded questions: ARC-E=${#ARC_E_Q[@]} ARC-C=${#ARC_C_Q[@]} HellaSwag=${#HS_Q[@]} MMLU=${#MMLU_Q[@]} GSM8K=${#GSM8K_Q[@]} TruthfulQA=${#TQA_Q[@]}"

# ── Hard check: every benchmark must have exactly 100 questions ───────
_q_ok=1
for _bench_arr in ARC_E_Q ARC_C_Q HS_Q MMLU_Q GSM8K_Q TQA_Q; do
    eval "_cnt=\${#${_bench_arr}[@]}"
    if [ "$_cnt" -ne 100 ]; then
        echo "ERROR: ${_bench_arr} has ${_cnt} questions (expected 100)."
        echo "       Delete ${_Q_CACHE} and re-run to regenerate the cache."
        _q_ok=0
    fi
done
[ "$_q_ok" -eq 0 ] && exit 1

# ── Section 7: Long Context / Needle-in-a-Haystack ────────────────────
# Kamradt (2023) methodology: hide a unique fact inside domain text;
# test retrieval at beginning / middle / end of context window.
# Liu et al. (2023) arXiv:2307.03172: retrieval degrades when needle is
# placed in the middle of a long context ("lost in the middle" effect).
#
# Evaluation: model response contains the exact string "XK-7749".
# Also measured: KV cache size (MB), prompt token fill %, RAM delta (MB).
NEEDLE_FACT="[SYSTEM NOTE] The primary authentication code for Project Helios is XK-7749."

# Haystack paragraphs — edge-AI domain, ~100 tokens each ──────────────
HP1="Edge computing distributes computational workloads closer to the data source, reducing latency and bandwidth consumption in IoT deployments. Rather than routing all sensor data to centralized cloud servers, edge nodes perform local inference and transmit only summarized results upstream. This approach benefits time-sensitive applications such as industrial control, autonomous navigation, and real-time health monitoring. Small language models running on embedded hardware enable natural language interfaces without cloud dependency."

HP2="The Raspberry Pi 5 features a quad-core ARM Cortex-A76 processor running at 2.4 GHz, offering substantially improved integer performance over its predecessors. With up to 8 GB of LPDDR4X memory and a PCIe 2.0 interface, the platform supports modest inference workloads using quantized neural network models. Thermal management is critical during sustained inference; the official active cooling solution maintains junction temperatures below 80 degrees Celsius under continuous load. USB 3.0 connectivity enables fast model transfer from host machines during benchmarking."

HP3="Transformer architectures rely on the self-attention mechanism to model token dependencies regardless of their distance in a sequence. The multi-head attention operation computes query, key, and value projections for each token, then takes a weighted sum of values based on query-key dot products. As sequence length grows, attention complexity scales quadratically, making long-context inference particularly expensive. Modern optimizations such as grouped-query attention and sliding-window attention reduce this overhead at the cost of modeling fidelity at extreme distances."

HP4="Post-training quantization reduces model weight precision from 16-bit floating point to lower bitwidths such as 4-bit or 2-bit integers. The GGUF format supports schemes including Q4_K_M, Q5_K_M, and Q8_0, each trading accuracy for reduced memory footprint and faster integer arithmetic. On CPUs without dedicated matrix acceleration, quantized kernels achieve throughput gains by fitting more weights in L2 and L3 cache. Calibration data quality significantly influences the accuracy retained after aggressive quantization to low bitwidths."

HP5="The MQTT protocol is widely adopted for lightweight publish-subscribe messaging in IoT deployments. Operating over TCP with a minimal two-byte fixed header, MQTT supports three quality-of-service levels: at-most-once, at-least-once, and exactly-once delivery semantics. Retained messages allow newly connected subscribers to immediately receive the last published value for a topic. Combined with TLS encryption and client certificate authentication, MQTT provides a secure and efficient transport layer for telemetry and command channels in distributed sensor networks."

HP6="The key-value cache in transformer inference stores previously computed attention keys and values for each layer, enabling efficient autoregressive generation without recomputing representations for prior tokens. Cache size scales with context length, number of layers, number of KV heads, and the head dimension. A 2048-token context with a 32-layer model using 8 KV heads of dimension 128 requires approximately 128 MB of contiguous memory. Pre-allocating this cache at server startup ensures deterministic memory usage and avoids fragmentation during extended sessions."

HP7="GGUF is a binary container format designed for efficient storage and loading of quantized language model weights. It supersedes the earlier GGML format by embedding all necessary metadata—tokenizer vocabulary, model hyperparameters, and quantization descriptors—within a single self-contained file. The format supports memory-mapped loading, allowing the operating system to page in only the weight tensors required for the current computation. This property is especially valuable on memory-constrained edge devices where loading an entire model into RAM would otherwise be prohibitive."

HP8="Federated learning trains a global model by aggregating gradient updates from multiple edge clients without centralizing raw data. Each participating device computes local gradients on its private dataset and transmits only the compressed update vector to a coordinating server. Differential privacy mechanisms add calibrated noise to individual updates, providing formal privacy guarantees. Communication efficiency techniques such as gradient sparsification and quantized communication reduce the bandwidth overhead inherent in large model updates across heterogeneous edge networks with limited uplink capacity."

HP9="Memory bandwidth is frequently the binding constraint for language model inference on consumer and embedded hardware. A 4-bit quantized 3-billion-parameter model requires approximately 1.5 GB of weight data; at 25 GB per second memory bandwidth the theoretical minimum time to read all weights once is about 60 milliseconds per token. ARM Cortex-A76 cores share a unified L3 cache of approximately 2 MB, insufficient to hold more than a small fraction of the weight matrices for even small models. Prefetching strategies and tiled matrix-vector multiplication layouts partially mitigate resulting cache misses."

LC_QUESTION="Based only on the information provided above, what is the authentication code for Project Helios? Reply with only the code, nothing else."

# Three composed prompts — needle position varies for recency/primacy/middle tests
LC1_PROMPT="${NEEDLE_FACT}

${HP1}

${HP2}

${HP3}

${LC_QUESTION}"

LC2_PROMPT="${HP1}

${HP2}

${HP3}

${HP4}

${NEEDLE_FACT}

${HP5}

${HP6}

${HP7}

${HP8}

${LC_QUESTION}"

LC3_PROMPT="${HP1}

${HP2}

${HP3}

${HP4}

${HP5}

${HP6}

${HP7}

${HP8}

${HP9}

${NEEDLE_FACT}

${LC_QUESTION}"

LC_SYSTEM="You are a precise reading-comprehension assistant. Read the provided text carefully and answer questions based solely on the information it contains."

# ═══════════════════════════════════════════════════════════════════════
#  Infrastructure / helper functions
# ═══════════════════════════════════════════════════════════════════════

track_step() {
    echo "[$(date '+%H:%M:%S')] model=${CURRENT_MODEL} section=${CURRENT_SECTION} q=${CURRENT_Q} step=$1" > "$STEP_FILE"
}

check_alive() {
    ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no \
        "$SSH_TARGET" "echo alive" 2>/dev/null | grep -q "alive"
}

assert_alive() {
    if ! check_alive; then
        local last; last=$(cat "$STEP_FILE" 2>/dev/null || echo "unknown")
        echo ""
        echo "╔══════════════════════════════════════════════════════════════════╗"
        echo "║  !! RPi OOM / CRASH DETECTED                                   ║"
        echo "╠══════════════════════════════════════════════════════════════════╣"
        printf  "║  Model   : %-54s║\n" "${CURRENT_MODEL}"
        printf  "║  Section : %-54s║\n" "${CURRENT_SECTION}"
        printf  "║  Question: %-54s║\n" "${CURRENT_Q}"
        printf  "║  Step    : %-54s║\n" "${last}"
        echo "╚══════════════════════════════════════════════════════════════════╝"
        { echo ""; echo "!!! OOM/CRASH: model=${CURRENT_MODEL} section=${CURRENT_SECTION} q=${CURRENT_Q}"; echo "    Last step: ${last}"; } >> "$REPORT_FILE" 2>/dev/null || true
        return 1
    fi
    return 0
}

start_heartbeat() {
    (
        while true; do
            sleep 5
            if ! ssh -o ConnectTimeout=4 -o BatchMode=yes -o StrictHostKeyChecking=no \
                    "$SSH_TARGET" "echo hb" 2>/dev/null | grep -q "hb"; then
                printf '\n[%s] !! HEARTBEAT LOST (model=%s section=%s) !!\n' \
                    "$(date '+%H:%M:%S')" "${CURRENT_MODEL}" "${CURRENT_SECTION}" >&2
                printf '[%s] Last step: %s\n' "$(date '+%H:%M:%S')" \
                    "$(cat "$STEP_FILE" 2>/dev/null || echo "unknown")" >&2
                break
            fi
        done
    ) &
    HEARTBEAT_PID=$!
}

stop_heartbeat() {
    [ -n "$HEARTBEAT_PID" ] && {
        kill "$HEARTBEAT_PID" 2>/dev/null || true
        wait "$HEARTBEAT_PID" 2>/dev/null || true
        HEARTBEAT_PID=""
    }
}

trap 'stop_heartbeat; stop_resource_sampler 2>/dev/null; echo ""; echo "EXIT — last step: $(cat "$STEP_FILE" 2>/dev/null)"' EXIT

# ── Resource sampler — polls llama-server CPU% and system RAM every 2s ─
# Each data line: "epoch cpu_pct ram_mb"
# Section markers: "MARK epoch section_name"
start_resource_sampler() {
    local model="$1"
    RESOURCE_SAMPLE_FILE=$(mktemp /tmp/res_sample_XXXXXX)
    # Write CSV header for the per-model timeseries file
    RESOURCE_CSV_FILE="standard_stress_${TS}_${model}_resources.csv"
    echo "timestamp,elapsed_s,section,cpu_pct,ram_used_mb" > "$RESOURCE_CSV_FILE"
    local t0; t0=$(date +%s)
    (
        while true; do
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
    # Store t0 so CSV writer knows elapsed seconds
    echo "T0=${t0}" >> "$RESOURCE_SAMPLE_FILE"
}

# Write a section boundary marker into the sample file
mark_section() {
    [ -n "$RESOURCE_SAMPLE_FILE" ] && \
        echo "MARK $(date +%s) $1" >> "$RESOURCE_SAMPLE_FILE"
}

# Stop the sampler, compute overall + per-section stats, write CSV, set globals
stop_resource_sampler() {
    if [ -n "$RESOURCE_SAMPLER_PID" ]; then
        kill "$RESOURCE_SAMPLER_PID" 2>/dev/null || true
        wait "$RESOURCE_SAMPLER_PID" 2>/dev/null || true
        RESOURCE_SAMPLER_PID=""
    fi
    [ ! -f "$RESOURCE_SAMPLE_FILE" ] && return

    # Parse timeseries and produce: overall stats + per-section table + CSV rows
    python3 - "$RESOURCE_SAMPLE_FILE" "$RESOURCE_CSV_FILE" << 'PYEOF'
import sys, re, os

sample_file = sys.argv[1]
csv_file    = sys.argv[2]

lines = open(sample_file).readlines()

# Extract T0
t0 = None
data = []    # (epoch, cpu, ram)
marks = []   # (epoch, section_name)

for line in lines:
    line = line.strip()
    if line.startswith('T0='):
        t0 = int(line[3:])
    elif line.startswith('MARK'):
        parts = line.split(None, 2)
        if len(parts) == 3:
            marks.append((int(parts[1]), parts[2]))
    else:
        parts = line.split()
        if len(parts) == 3:
            try:
                data.append((int(parts[0]), float(parts[1]), int(parts[2])))
            except ValueError:
                pass

if not t0:
    t0 = data[0][0] if data else 0

# Assign section label to each sample based on MARK boundaries
def section_for(ts):
    label = "init"
    for mk_ts, mk_name in marks:
        if ts >= mk_ts:
            label = mk_name
    return label

# Write CSV
with open(csv_file, 'a') as f:
    for epoch, cpu, ram in data:
        elapsed = epoch - t0
        section = section_for(epoch)
        f.write(f"{epoch},{elapsed},{section},{cpu:.1f},{ram}\n")

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

# Per-section stats
section_data = {}
for epoch, cpu, ram in data:
    s = section_for(epoch)
    section_data.setdefault(s, {'cpu': [], 'ram': [], 'ts': []})
    section_data[s]['cpu'].append(cpu)
    section_data[s]['ram'].append(ram)
    section_data[s]['ts'].append(epoch)

# Print in order of first appearance
seen = []
for epoch, cpu, ram in data:
    s = section_for(epoch)
    if s not in seen:
        seen.append(s)

for s in seen:
    d = section_data[s]
    cpu_avg = sum(d['cpu'])/len(d['cpu']) if d['cpu'] else 0
    cpu_pk  = max(d['cpu']) if d['cpu'] else 0
    ram_avg = int(sum(d['ram'])/len(d['ram'])) if d['ram'] else 0
    ram_pk  = max(d['ram']) if d['ram'] else 0
    dur     = d['ts'][-1] - d['ts'][0] if len(d['ts']) > 1 else 0
    print(f"SECTION|{s}|{cpu_avg:.1f}|{cpu_pk:.1f}|{ram_avg}|{ram_pk}|{dur}")
PYEOF

    rm -f "$RESOURCE_SAMPLE_FILE"
    RESOURCE_SAMPLE_FILE=""
}

# Parse stop_resource_sampler output into globals and log the per-section table
parse_and_log_resources() {
    local raw_output="$1"
    local model="$2"

    CPU_AVG="N/A"; CPU_PEAK="N/A"; RAM_AVG="N/A"; RAM_PEAK="N/A"

    while IFS= read -r line; do
        case "$line" in
            OVERALL_CPU_AVG=*)  CPU_AVG="${line#*=}"  ;;
            OVERALL_CPU_PEAK=*) CPU_PEAK="${line#*=}" ;;
            OVERALL_RAM_AVG=*)  RAM_AVG="${line#*=}"  ;;
            OVERALL_RAM_PEAK=*) RAM_PEAK="${line#*=}" ;;
            SECTION|*)
                IFS='|' read -r _ sec_name cpu_a cpu_p ram_a ram_p dur <<< "$line"
                local dur_fmt; dur_fmt=$(fmt_time "$dur")
                log "$(printf '    %-14s │ %7s%% │ %8s%% │ %7s MB │ %8s MB │ %s' \
                    "$sec_name" "$cpu_a" "$cpu_p" "$ram_a" "$ram_p" "$dur_fmt")"
                ;;
        esac
    done <<< "$raw_output"

    log "$(printf '    %-14s │ %7s%% │ %8s%% │ %7s MB │ %8s MB │ %s' \
        'OVERALL' "$CPU_AVG" "$CPU_PEAK" "$RAM_AVG" "$RAM_PEAK" "—")"

    R_CPU_AVG["$model"]="$CPU_AVG"
    R_CPU_PEAK["$model"]="$CPU_PEAK"
    R_RAM_AVG["$model"]="$RAM_AVG"
    R_RAM_PEAK["$model"]="$RAM_PEAK"

    log "  Timeseries CSV → ${RESOURCE_CSV_FILE}"
}

wait_for_server() {
    local elapsed=0
    echo -n "    Waiting for llama-server"
    until curl -s "http://${SSH_HOST}:${SERVER_PORT}/health" 2>/dev/null | grep -q "ok"; do
        sleep 2; elapsed=$((elapsed + 2)); echo -n "."
        if [ "$elapsed" -ge "$SERVER_READY_TIMEOUT" ]; then echo " TIMEOUT"; return 1; fi
    done
    echo " ready (${elapsed}s)"
    return 0
}

clean_edge() {
    ssh "$SSH_TARGET" "
        pkill -f llama-server 2>/dev/null || true
        cd ~/projects/raspi-claw 2>/dev/null && docker compose down 2>/dev/null || true
    " 2>/dev/null || true
    sleep 4
}

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

fmt_time() { printf "%dm%02ds" $(( $1 / 60 )) $(( $1 % 60 )); }

# ── Restart llama-server in-place (mid-section OOM recovery) ──────────
# Uses globals: MODEL_FILE, REMOTE_DIR, SERVER_PORT, CTX_SIZE, THREADS, N_PREDICT_MATH
# Returns 0 on success, 1 if server fails to come back.
#
# Recovery sequence:
#   1. SIGTERM → SIGKILL llama-server
#   2. Flush OS page cache (sync + drop_caches) — reclaims 100–200 MB on Pi
#   3. Poll until used RAM drops below RAM_RESTART_THRESHOLD_MB (or 30s timeout)
#   4. Relaunch and wait for /health
restart_server_for_batch() {
    local reason="$1"   # "proactive" | "reactive" | "ram-pressure"
    local ram_before; ram_before=$(measure_ram_mb)
    log "    [OOM-guard/${reason}] Stopping server  (RAM was ${ram_before} MB)..."

    ssh "$SSH_TARGET" "
        pkill -SIGTERM -f llama-server 2>/dev/null || true
        sleep 3
        pkill -SIGKILL -f llama-server 2>/dev/null || true
        sleep 2
        # Flush OS page/dentry/inode caches to reclaim model weight pages
        sync 2>/dev/null || true
        sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || \
            sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
        sleep 1
    " 2>/dev/null || true

    # Poll until RAM drops to a safe level (max 30 s)
    local poll=0
    local safe_ram=$(( ${RAM_RESTART_THRESHOLD_MB:-1450} - 50 ))
    while [ "$poll" -lt 30 ]; do
        local cur_ram; cur_ram=$(measure_ram_mb)
        if [ "${cur_ram:-9999}" -le "$safe_ram" ]; then
            log "    [OOM-guard/${reason}] RAM freed: ${ram_before} → ${cur_ram} MB — relaunching"
            break
        fi
        sleep 2; poll=$(( poll + 2 ))
    done
    if [ "$poll" -ge 30 ]; then
        local cur_ram; cur_ram=$(measure_ram_mb)
        log "    [OOM-guard/${reason}] RAM poll timeout (still ${cur_ram} MB) — launching anyway"
    fi

    ssh "$SSH_TARGET" "bash -c '
        cd ${REMOTE_DIR} || exit 1
        nohup ./llama-server \
            -m model/${MODEL_FILE} \
            --host 0.0.0.0 \
            --port ${SERVER_PORT} \
            --ctx-size ${CTX_SIZE} \
            --threads ${THREADS} \
            -n ${N_PREDICT_MATH} \
            >> llama-server.log 2>&1 &
        disown \$! 2>/dev/null || true
        echo started
    '" 2>/dev/null || true

    if ! wait_for_server; then
        log "    [OOM-guard/${reason}] !! Server failed to restart — marking section as failed"
        return 1
    fi
    local ram_after; ram_after=$(measure_ram_mb)
    log "    [OOM-guard/${reason}] Server ready  (RAM now ${ram_after} MB) — resuming"
    return 0
}

# ── KV cache size — parse llama-server.log after startup ─────────────
# llama.cpp logs: "llm_load_tensors: kv self size  =  NNN.NN MiB"
# Returns the numeric MiB value, or "N/A" on failure.
get_kv_cache_mb() {
    ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no \
        "$SSH_TARGET" \
        "grep -i 'kv self size\|kv buffer\|KV buffer' \
         ${REMOTE_DIR}/llama-server.log 2>/dev/null | \
         grep -oP '[\d.]+(?=\s*MiB)' | head -1" 2>/dev/null || echo "N/A"
}

# ── Instantaneous RAM snapshot (used MB) on edge device ──────────────
measure_ram_mb() {
    ssh -o ConnectTimeout=3 -o BatchMode=yes -o StrictHostKeyChecking=no \
        "$SSH_TARGET" \
        "free -m | awk '/^Mem:/{print \$3}'" 2>/dev/null || echo "0"
}

# ── Single-call CPU% + RAM snapshot — returns "cpu_pct ram_mb" ────────
# Combines ps %cpu for llama-server and free -m into one SSH round-trip.
snapshot_resources() {
    ssh -o ConnectTimeout=3 -o BatchMode=yes -o StrictHostKeyChecking=no \
        "$SSH_TARGET" \
        "cpu=\$(ps -C llama-server -o %cpu= 2>/dev/null | awk '{s+=\$1}END{printf \"%.1f\",s+0}'); \
         ram=\$(free -m | awk '/^Mem:/{print \$3}'); \
         echo \"\${cpu:-0} \${ram:-0}\"" 2>/dev/null || echo "0 0"
}

# ── API call wrapper ──────────────────────────────────────────────────
# Sets LAST_RESPONSE, LAST_LATENCY_MS, LAST_PROMPT_TOKENS, LAST_COMPLETION_TOKENS
LAST_RESPONSE=""
LAST_LATENCY_MS=0
LAST_PROMPT_TOKENS=0
LAST_COMPLETION_TOKENS=0

call_api() {
    local system_prompt="$1"
    local user_prompt="$2"
    local max_tokens="$3"

    local t0; t0=$(date +%s%3N)
    local raw
    raw=$(curl -s --max-time 600 \
        "http://${SSH_HOST}:${SERVER_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$(python3 -c "
import json, sys
print(json.dumps({
    'model': 'local',
    'messages': [
        {'role': 'system', 'content': sys.argv[1]},
        {'role': 'user',   'content': sys.argv[2]}
    ],
    'max_tokens': int(sys.argv[3]),
    'temperature': 0.0
}))
" "$system_prompt" "$user_prompt" "$max_tokens" 2>/dev/null)" 2>/dev/null || echo "")

    local t1; t1=$(date +%s%3N)
    LAST_LATENCY_MS=$(( t1 - t0 ))
    LAST_RESPONSE=$(echo "$raw" | python3 -c "
import sys, json
try:
    print(json.load(sys.stdin)['choices'][0]['message']['content'])
except:
    print('ERROR')
" 2>/dev/null || echo "ERROR")
    LAST_PROMPT_TOKENS=$(echo "$raw" | python3 -c "
import sys, json
try:
    print(json.load(sys.stdin).get('usage', {}).get('prompt_tokens', 0))
except:
    print(0)
" 2>/dev/null || echo "0")
    LAST_COMPLETION_TOKENS=$(echo "$raw" | python3 -c "
import sys, json
try:
    print(json.load(sys.stdin).get('usage', {}).get('completion_tokens', 0))
except:
    print(0)
" 2>/dev/null || echo "0")
}

# ── Multiple-choice answer extractor ─────────────────────────────────
# Parses the model response and extracts A/B/C/D.
# Strategy (in priority order):
#   1. "Answer: X" or "answer is X" pattern
#   2. Standalone letter at start of response
#   3. First A/B/C/D found anywhere in response
parse_mc() {
    local response="$1"
    python3 - "$response" << 'PYEOF'
import sys, re
text = sys.argv[1].strip()
# Priority 1: explicit answer markers
for pat in [
    r'[Aa]nswer[:\s]+\(?([A-D])\)?',
    r'[Tt]he answer is[:\s]+\(?([A-D])\)?',
    r'[Cc]orrect answer[:\s]+\(?([A-D])\)?',
    r'^\s*\(?([A-D])\)?[\s\.\)]',
]:
    m = re.search(pat, text, re.MULTILINE)
    if m:
        print(m.group(1).upper())
        sys.exit(0)
# Priority 2: first standalone A/B/C/D letter anywhere
m = re.search(r'\b([A-D])\b', text)
if m:
    print(m.group(1).upper())
    sys.exit(0)
print("?")
PYEOF
}

# ── GSM8K numeric answer extractor ───────────────────────────────────
# Looks for "#### NUMBER" (standard GSM8K format), then last integer in response.
parse_number() {
    local response="$1"
    python3 - "$response" << 'PYEOF'
import sys, re
text = sys.argv[1]
# Priority 1: GSM8K standard "#### N" marker
m = re.search(r'####\s*([\d,]+\.?\d*)', text)
if m:
    print(m.group(1).replace(',', '').rstrip('.'))
    sys.exit(0)
# Priority 2: "answer is N" or "= N" at end of sentence
for pat in [r'[Aa]nswer[:\s]+\$?([\d,]+)', r'=\s*([\d,]+)\s*\.?\s*$']:
    m = re.search(pat, text, re.MULTILINE)
    if m:
        print(m.group(1).replace(',', ''))
        sys.exit(0)
# Priority 3: last standalone integer in the response
nums = re.findall(r'\b(\d+)\b', text)
print(nums[-1] if nums else "?")
PYEOF
}

# ── Run one multiple-choice section ──────────────────────────────────
# Args: section_name system_prompt questions_array answers_array
# Sets: SECTION_CORRECT SECTION_TOTAL SECTION_TIME_S
SECTION_CORRECT=0; SECTION_TOTAL=0; SECTION_TIME_S=0

run_mc_section() {
    local section="$1"
    local sys_prompt="$2"
    local -n _qs="$3"
    local -n _ans="$4"

    SECTION_CORRECT=0; SECTION_TOTAL=0
    local sec_start; sec_start=$(date +%s)
    local section_restarts=0
    local section_aborted=0
    CURRENT_SECTION="$section"

    for i in "${!_qs[@]}"; do
        local qnum=$((i+1))
        CURRENT_Q="Q${qnum}"
        track_step "api_call"

        # ── Proactive restart: every BATCH_RESTART questions ─────────────
        if [ "${BATCH_RESTART:-0}" -gt 0 ] && [ "$qnum" -gt 1 ] && \
           [ $(( (qnum - 1) % BATCH_RESTART )) -eq 0 ]; then
            if ! restart_server_for_batch "proactive"; then
                section_aborted=1; break
            fi
            section_restarts=$(( section_restarts + 1 ))
        fi

        local q_ts; q_ts=$(date '+%Y-%m-%d %H:%M:%S')
        local snap_pre; snap_pre=$(snapshot_resources)
        local cpu_pre; cpu_pre=$(echo "$snap_pre" | awk '{print $1}')
        local ram_pre; ram_pre=$(echo "$snap_pre" | awk '{print $2}')

        # ── RAM-threshold restart: preempt OOM before the API call ────────
        if [ "${RAM_RESTART_THRESHOLD_MB:-0}" -gt 0 ] && \
           [ "${ram_pre:-0}" -ge "${RAM_RESTART_THRESHOLD_MB}" ]; then
            log "    [OOM-guard/ram-pressure] RAM=${ram_pre}MB ≥ threshold ${RAM_RESTART_THRESHOLD_MB}MB at Q${qnum} — preemptive restart"
            section_restarts=$(( section_restarts + 1 ))
            if ! restart_server_for_batch "ram-pressure"; then
                section_aborted=1; break
            fi
            # Refresh snapshot after restart
            snap_pre=$(snapshot_resources)
            cpu_pre=$(echo "$snap_pre" | awk '{print $1}')
            ram_pre=$(echo "$snap_pre" | awk '{print $2}')
        fi

        call_api "$sys_prompt" "$(printf '%b' "${_qs[$i]}")" "$N_PREDICT_MC"

        # ── Reactive restart: server died mid-section ─────────────────────
        if [ "$LAST_RESPONSE" = "ERROR" ] && ! check_alive; then
            log "    [OOM-guard/reactive] Server unresponsive at Q${qnum} — attempting restart..."
            section_restarts=$(( section_restarts + 1 ))
            if ! restart_server_for_batch "reactive"; then
                section_aborted=1; break
            fi
            # Retry the question once after recovery
            call_api "$sys_prompt" "$(printf '%b' "${_qs[$i]}")" "$N_PREDICT_MC"
        fi

        local snap_post; snap_post=$(snapshot_resources)
        local cpu_post; cpu_post=$(echo "$snap_post" | awk '{print $1}')
        local ram_post; ram_post=$(echo "$snap_post" | awk '{print $2}')

        local predicted; predicted=$(parse_mc "$LAST_RESPONSE")
        local correct="${_ans[$i]}"
        local ok=0; [ "$predicted" = "$correct" ] && ok=1
        SECTION_CORRECT=$((SECTION_CORRECT + ok))
        SECTION_TOTAL=$((SECTION_TOTAL + 1))

        local marker; marker=$([ "$ok" -eq 1 ] && echo "✓" || echo "✗")
        log "    Q${qnum} [${q_ts}]: predicted=${predicted} correct=${correct} ${marker}  latency=${LAST_LATENCY_MS}ms  cpu=${cpu_pre}→${cpu_post}%  ram=${ram_pre}→${ram_post}MB"
        log "         Response: $(echo "$LAST_RESPONSE" | head -1 | cut -c1-100)"
    done

    local sec_end; sec_end=$(date +%s)
    SECTION_TIME_S=$(( sec_end - sec_start ))
    local pct="N/A"
    [ "$SECTION_TOTAL" -gt 0 ] && pct=$(python3 -c "print(f'{$SECTION_CORRECT / $SECTION_TOTAL * 100:.0f}')")
    local restart_note=""
    [ "$section_restarts" -gt 0 ] && restart_note="  [${section_restarts} server restart(s)]"
    [ "$section_aborted" -eq 1 ] && restart_note="${restart_note}  [ABORTED after restart failure]"
    log "  → ${section}: ${SECTION_CORRECT}/${SECTION_TOTAL} = ${pct}%  ($(fmt_time $SECTION_TIME_S))${restart_note}"
    return "$section_aborted"
}

# ── Run GSM8K section ─────────────────────────────────────────────────
run_gsm8k_section() {
    local -n _qs="$1"
    local -n _ans="$2"

    SECTION_CORRECT=0; SECTION_TOTAL=0
    local sec_start; sec_start=$(date +%s)
    local section_restarts=0
    local section_aborted=0
    CURRENT_SECTION="GSM8K"

    for i in "${!_qs[@]}"; do
        local qnum=$((i+1))
        CURRENT_Q="GSM_Q${qnum}"
        track_step "api_call"

        # ── Proactive restart every BATCH_RESTART questions ───────────────
        # GSM8K generates up to 512 tokens per question — highest RAM pressure of all sections.
        if [ "${BATCH_RESTART:-0}" -gt 0 ] && [ "$qnum" -gt 1 ] && \
           [ $(( (qnum - 1) % BATCH_RESTART )) -eq 0 ]; then
            if ! restart_server_for_batch "proactive"; then
                section_aborted=1; break
            fi
            section_restarts=$(( section_restarts + 1 ))
        fi

        local q_ts; q_ts=$(date '+%Y-%m-%d %H:%M:%S')
        local snap_pre; snap_pre=$(snapshot_resources)
        local cpu_pre; cpu_pre=$(echo "$snap_pre" | awk '{print $1}')
        local ram_pre; ram_pre=$(echo "$snap_pre" | awk '{print $2}')

        # ── RAM-threshold restart: preempt OOM before the API call ────────
        if [ "${RAM_RESTART_THRESHOLD_MB:-0}" -gt 0 ] && \
           [ "${ram_pre:-0}" -ge "${RAM_RESTART_THRESHOLD_MB}" ]; then
            log "    [OOM-guard/ram-pressure] RAM=${ram_pre}MB ≥ threshold ${RAM_RESTART_THRESHOLD_MB}MB at GSM_Q${qnum} — preemptive restart"
            section_restarts=$(( section_restarts + 1 ))
            if ! restart_server_for_batch "ram-pressure"; then
                section_aborted=1; break
            fi
            snap_pre=$(snapshot_resources)
            cpu_pre=$(echo "$snap_pre" | awk '{print $1}')
            ram_pre=$(echo "$snap_pre" | awk '{print $2}')
        fi

        call_api "$GSM8K_SYSTEM" "${_qs[$i]}" "$N_PREDICT_MATH"

        # ── Reactive restart: server died mid-section ─────────────────────
        if [ "$LAST_RESPONSE" = "ERROR" ] && ! check_alive; then
            log "    [OOM-guard/reactive] Server unresponsive at GSM_Q${qnum} — attempting restart..."
            section_restarts=$(( section_restarts + 1 ))
            if ! restart_server_for_batch "reactive"; then
                section_aborted=1; break
            fi
            call_api "$GSM8K_SYSTEM" "${_qs[$i]}" "$N_PREDICT_MATH"
        fi

        local snap_post; snap_post=$(snapshot_resources)
        local cpu_post; cpu_post=$(echo "$snap_post" | awk '{print $1}')
        local ram_post; ram_post=$(echo "$snap_post" | awk '{print $2}')

        local predicted; predicted=$(parse_number "$LAST_RESPONSE")
        local correct="${_ans[$i]}"
        local ok=0; [ "$predicted" = "$correct" ] && ok=1
        SECTION_CORRECT=$((SECTION_CORRECT + ok))
        SECTION_TOTAL=$((SECTION_TOTAL + 1))

        local marker; marker=$([ "$ok" -eq 1 ] && echo "✓" || echo "✗")
        log "    Q${qnum} [${q_ts}]: predicted=${predicted} correct=${correct} ${marker}  latency=${LAST_LATENCY_MS}ms  cpu=${cpu_pre}→${cpu_post}%  ram=${ram_pre}→${ram_post}MB"
        log "         Response: $(echo "$LAST_RESPONSE" | tail -3 | tr '\n' ' ' | cut -c1-120)"
    done

    local sec_end; sec_end=$(date +%s)
    SECTION_TIME_S=$(( sec_end - sec_start ))
    local pct="N/A"
    [ "$SECTION_TOTAL" -gt 0 ] && pct=$(python3 -c "print(f'{$SECTION_CORRECT / $SECTION_TOTAL * 100:.0f}')")
    local restart_note=""
    [ "$section_restarts" -gt 0 ] && restart_note="  [${section_restarts} server restart(s)]"
    [ "$section_aborted" -eq 1 ] && restart_note="${restart_note}  [ABORTED after restart failure]"
    log "  → GSM8K: ${SECTION_CORRECT}/${SECTION_TOTAL} = ${pct}%  ($(fmt_time $SECTION_TIME_S))${restart_note}"
    return "$section_aborted"
}

# ── Run Section 7: Long Context / Needle-in-a-Haystack ───────────────
# Tests: LC1 (needle at beginning), LC2 (middle), LC3 (end).
# Sets: LC_PCT (0/33/67/100%), LC_TIME_S, LC_DETAIL (per-prompt log line)
LC_PCT="N/A"; LC_TIME_S=0; LC_DETAIL=""

run_long_context_section() {
    CURRENT_SECTION="LongContext"
    local lc_correct=0
    local sec_start; sec_start=$(date +%s)

    local lc_labels=("LC1_begin" "LC2_middle" "LC3_end")
    local lc_prompts=("$LC1_PROMPT" "$LC2_PROMPT" "$LC3_PROMPT")
    LC_DETAIL=""

    for i in 0 1 2; do
        local label="${lc_labels[$i]}"
        CURRENT_Q="$label"
        track_step "api_call"

        local q_ts; q_ts=$(date '+%Y-%m-%d %H:%M:%S')
        local snap_pre; snap_pre=$(snapshot_resources)
        local cpu_pre; cpu_pre=$(echo "$snap_pre" | awk '{print $1}')
        local ram_before; ram_before=$(echo "$snap_pre" | awk '{print $2}')

        call_api "$LC_SYSTEM" "${lc_prompts[$i]}" "$N_PREDICT_LC"

        # ── Reactive restart: server died on large context prompt ─────────
        if [ "$LAST_RESPONSE" = "ERROR" ] && ! check_alive; then
            log "    [OOM-guard/reactive] Server unresponsive at ${label} — attempting restart..."
            if restart_server_for_batch "reactive"; then
                call_api "$LC_SYSTEM" "${lc_prompts[$i]}" "$N_PREDICT_LC"
            fi
        fi

        local snap_post; snap_post=$(snapshot_resources)
        local cpu_post; cpu_post=$(echo "$snap_post" | awk '{print $1}')
        local ram_after; ram_after=$(echo "$snap_post" | awk '{print $2}')
        local ram_delta=$(( ram_after - ram_before ))

        local found=0
        echo "$LAST_RESPONSE" | grep -qF "XK-7749" && found=1
        lc_correct=$(( lc_correct + found ))

        local marker; marker=$([ "$found" -eq 1 ] && echo "FOUND" || echo "MISS")
        local ctx_fill="?"
        if [ "${LAST_PROMPT_TOKENS:-0}" -gt 0 ]; then
            ctx_fill=$(python3 -c "print(f'{${LAST_PROMPT_TOKENS}/${CTX_SIZE}*100:.0f}')")
        fi
        local detail_line
        detail_line="    ${label} [${q_ts}]: needle=${marker}  latency=${LAST_LATENCY_MS}ms  ctx_fill=${ctx_fill}%  cpu=${cpu_pre}→${cpu_post}%  ram=${ram_before}→${ram_after}MB (delta=${ram_delta}MB)  prompt_tok=${LAST_PROMPT_TOKENS:-?}  compl_tok=${LAST_COMPLETION_TOKENS:-?}"
        log "$detail_line"
        log "         Response: $(echo "$LAST_RESPONSE" | head -1 | cut -c1-100)"
        LC_DETAIL="${LC_DETAIL}${detail_line}"$'\n'
    done

    local sec_end; sec_end=$(date +%s)
    LC_TIME_S=$(( sec_end - sec_start ))
    LC_PCT=$(python3 -c "print(f'{${lc_correct}/3*100:.0f}')")
    log "  → LongContext (Needle): ${lc_correct}/3 = ${LC_PCT}%  ($(fmt_time $LC_TIME_S))"
    log "    Ref: Kamradt (2023) NeedleInHaystack; Liu et al. (2023) arXiv:2307.03172"
}

# ─────────────────────────────────────────────────────────────────────
#  Result stores
# ─────────────────────────────────────────────────────────────────────
declare -A R_ARC_E R_ARC_C R_HELLASWAG R_MMLU R_GSM8K R_TQA R_LONGCTX
declare -A R_TIME R_STATUS R_COMPOSITE
declare -A R_CPU_AVG R_CPU_PEAK R_RAM_AVG R_RAM_PEAK
declare -A R_KV_MB
OOM_MODELS=()
DONE_MODELS=()

# System prompts for MC sections
MC_SYSTEM_0SHOT="You are a knowledgeable assistant. Read the question carefully and answer with only the letter of the correct choice: A, B, C, or D. Do not explain."

# ═══════════════════════════════════════════════════════════════════════
#  Main benchmark loop
# ═══════════════════════════════════════════════════════════════════════
sep2
log " STANDARD SLM BENCHMARK"
log " Generated  : $(date '+%Y-%m-%d %H:%M:%S')"
log " Target     : ${SSH_TARGET}"
log " Models     : ${#MODELS[@]}"
log " Sections   : ARC-Easy[1] ARC-Challenge[1] HellaSwag[2] MMLU(5-shot)[3] GSM8K(2-shot)[4] TruthfulQA[5] LongCtx[7]"
log " Protocol   : 0-shot MC ARC/HellaSwag/TruthfulQA | 5-shot MMLU | 2-shot CoT GSM8K | Needle@3pos LongCtx"
sep2
log ""

BENCH_START=$(date +%s)
MDL_IDX=0

for model_entry in "${MODELS[@]}"; do
    MDL_IDX=$((MDL_IDX + 1))
    HF_REPO=$(echo "$model_entry"     | cut -d'|' -f1)
    MODEL_FILE=$(echo "$model_entry"  | cut -d'|' -f2)
    PICOCLAW_CFG=$(echo "$model_entry"| cut -d'|' -f3)
    MODEL_PATH="${LOCAL_MODEL_DIR}/${MODEL_FILE}"
    QUANT=$(echo "$MODEL_FILE" | grep -oE 'Q[0-9]+_[A-Z0-9_]+|Q[0-9]+_[0-9]+' | head -1 || echo "?")
    CURRENT_MODEL="$PICOCLAW_CFG"
    MDL_START=$(date +%s)

    # ── Liveness gate ──────────────────────────────────────────────────
    CURRENT_SECTION="pre_flight"; CURRENT_Q="liveness_check"
    track_step "check_alive"
    if ! check_alive; then
        echo "╔══════════════════════════════════════════════════════════════════╗"
        echo "║  !! RPi UNREACHABLE — cannot start model                       ║"
        printf  "║  Model : %-57s║\n" "$PICOCLAW_CFG"
        echo "╚══════════════════════════════════════════════════════════════════╝"
        R_STATUS["$PICOCLAW_CFG"]="unreachable"
        OOM_MODELS+=("${PICOCLAW_CFG} [UNREACHABLE]")
        continue
    fi

    track_step "clean_edge"
    log "  Pre-flight cleanup..."
    clean_edge; log "  Pre-flight done."

    sep2
    log "[${MDL_IDX}/${#MODELS[@]}] ${PICOCLAW_CFG}  (${QUANT})"
    log "  File : ${MODEL_FILE}"
    sep2

    # ── Transfer model ─────────────────────────────────────────────────
    CURRENT_SECTION="setup"; CURRENT_Q="model_transfer"
    track_step "check_remote"
    ssh "$SSH_TARGET" "mkdir -p ${REMOTE_MODEL_DIR}"
    if ssh "$SSH_TARGET" "[ -f ${REMOTE_MODEL_DIR}/${MODEL_FILE} ]" 2>/dev/null; then
        log "  Model already on edge."
    else
        if [ ! -f "$MODEL_PATH" ]; then
            log "  Downloading from HuggingFace..."
            mkdir -p "$LOCAL_MODEL_DIR"
            if ! huggingface-cli download "$HF_REPO" "$MODEL_FILE" --local-dir "$LOCAL_MODEL_DIR"; then
                log "  ERROR: Download failed — skipping."
                R_STATUS["$PICOCLAW_CFG"]="download_fail"
                continue
            fi
        fi
        track_step "scp_transfer"
        log "  Transferring to edge (this may take a few minutes)..."
        scp "$MODEL_PATH" "${SSH_TARGET}:${REMOTE_MODEL_DIR}/"
        log "  Transfer done."
    fi

    # ── Stop old server, start new one ────────────────────────────────
    CURRENT_SECTION="server"; CURRENT_Q="start"
    track_step "stop_old_server"
    log "  Stopping any existing server..."
    stop_server
    log "  Launching llama-server..."
    track_step "launch_server"
    ssh "$SSH_TARGET" "bash -c '
        cd ${REMOTE_DIR} || exit 1
        nohup ./llama-server \
            -m model/${MODEL_FILE} \
            --host 0.0.0.0 \
            --port ${SERVER_PORT} \
            --ctx-size ${CTX_SIZE} \
            --threads ${THREADS} \
            -n ${N_PREDICT_MATH} \
            > llama-server.log 2>&1 &
        disown \$! 2>/dev/null || true
        echo started
    '" || true

    track_step "wait_ready"
    if ! wait_for_server; then
        log "  ERROR: Server timed out — skipping."
        R_STATUS["$PICOCLAW_CFG"]="server_fail"
        OOM_MODELS+=("${PICOCLAW_CFG} [SERVER FAIL]")
        clean_edge; continue
    fi

    start_heartbeat
    start_resource_sampler "$PICOCLAW_CFG"
    log "    Resource sampler started (CPU + RAM every 2s → ${RESOURCE_CSV_FILE})"

    # ── KV cache pre-allocation (from server startup log) ────────────
    CURRENT_SECTION="setup"; CURRENT_Q="kv_cache_size"
    kv_mb=$(get_kv_cache_mb)
    if [ "$kv_mb" != "N/A" ] && [ -n "$kv_mb" ]; then
        log "    KV cache pre-allocated: ${kv_mb} MiB  (ctx=${CTX_SIZE} tokens)"
    else
        kv_mb="N/A"
        log "    KV cache size: N/A (not found in server log)"
    fi
    R_KV_MB["$PICOCLAW_CFG"]="$kv_mb"

    # ── local result vars ──────────────────────────────────────────────
    arc_e_c=0;  arc_e_t=0;  arc_e_pct="N/A"; arc_e_s=0
    arc_c_c=0;  arc_c_t=0;  arc_c_pct="N/A"; arc_c_s=0
    hs_c=0;     hs_t=0;     hs_pct="N/A";    hs_s=0
    mmlu_c=0;   mmlu_t=0;   mmlu_pct="N/A";  mmlu_s=0
    gsm_c=0;    gsm_t=0;    gsm_pct="N/A";   gsm_s=0
    tqa_c=0;    tqa_t=0;    tqa_pct="N/A";   tqa_s=0
    lc_pct="N/A"; lc_s=0
    kv_mb="N/A"
    oom_at=""

    # ════════════════════════════════════════════════════════════════════
    #  [1] ARC-Easy  —  0-shot, normalized accuracy  [Clark et al. 2018]
    # ════════════════════════════════════════════════════════════════════
    sep
    log "  [1/7] ARC-Easy  (0-shot, 100 questions)  [Clark et al. 2018]"
    mark_section "ARC-Easy"
    run_mc_section "ARC-Easy" "$MC_SYSTEM_0SHOT" ARC_E_Q ARC_E_ANS
    arc_e_c=$SECTION_CORRECT; arc_e_t=$SECTION_TOTAL; arc_e_s=$SECTION_TIME_S
    arc_e_pct="N/A"
    [ "$arc_e_t" -gt 0 ] && arc_e_pct=$(python3 -c "print(f'{$arc_e_c/$arc_e_t*100:.0f}')")
    R_ARC_E["$PICOCLAW_CFG"]="$arc_e_pct"

    if ! assert_alive; then
        stop_resource_sampler 2>/dev/null; stop_heartbeat; oom_at="ARC-Easy"
        R_STATUS["$PICOCLAW_CFG"]="oom@ARC-Easy"
        OOM_MODELS+=("${PICOCLAW_CFG} [OOM at ARC-Easy]")
        MDL_END=$(date +%s); R_TIME["$PICOCLAW_CFG"]=$(( MDL_END - MDL_START ))
        clean_edge; continue
    fi

    # ════════════════════════════════════════════════════════════════════
    #  [2] ARC-Challenge  —  0-shot  [Clark et al. 2018]
    # ════════════════════════════════════════════════════════════════════
    sep
    log "  [2/7] ARC-Challenge  (0-shot, 100 questions)  [Clark et al. 2018]"
    mark_section "ARC-Challenge"
    run_mc_section "ARC-Challenge" "$MC_SYSTEM_0SHOT" ARC_C_Q ARC_C_ANS
    arc_c_c=$SECTION_CORRECT; arc_c_t=$SECTION_TOTAL; arc_c_s=$SECTION_TIME_S
    arc_c_pct="N/A"
    [ "$arc_c_t" -gt 0 ] && arc_c_pct=$(python3 -c "print(f'{$arc_c_c/$arc_c_t*100:.0f}')")
    R_ARC_C["$PICOCLAW_CFG"]="$arc_c_pct"

    if ! assert_alive; then
        stop_resource_sampler 2>/dev/null; stop_heartbeat; oom_at="ARC-Challenge"
        R_STATUS["$PICOCLAW_CFG"]="oom@ARC-C"
        OOM_MODELS+=("${PICOCLAW_CFG} [OOM at ARC-Challenge]")
        MDL_END=$(date +%s); R_TIME["$PICOCLAW_CFG"]=$(( MDL_END - MDL_START ))
        clean_edge; continue
    fi

    # ════════════════════════════════════════════════════════════════════
    #  [3] HellaSwag  —  0-shot  [Zellers et al. 2019]
    # ════════════════════════════════════════════════════════════════════
    sep
    log "  [3/7] HellaSwag  (0-shot, 100 questions)  [Zellers et al. 2019]"
    mark_section "HellaSwag"
    run_mc_section "HellaSwag" "$MC_SYSTEM_0SHOT" HS_Q HS_ANS
    hs_c=$SECTION_CORRECT; hs_t=$SECTION_TOTAL; hs_s=$SECTION_TIME_S
    hs_pct="N/A"
    [ "$hs_t" -gt 0 ] && hs_pct=$(python3 -c "print(f'{$hs_c/$hs_t*100:.0f}')")
    R_HELLASWAG["$PICOCLAW_CFG"]="$hs_pct"

    if ! assert_alive; then
        stop_resource_sampler 2>/dev/null; stop_heartbeat; oom_at="HellaSwag"
        R_STATUS["$PICOCLAW_CFG"]="oom@HellaSwag"
        OOM_MODELS+=("${PICOCLAW_CFG} [OOM at HellaSwag]")
        MDL_END=$(date +%s); R_TIME["$PICOCLAW_CFG"]=$(( MDL_END - MDL_START ))
        clean_edge; continue
    fi

    # ════════════════════════════════════════════════════════════════════
    #  [4] MMLU  —  5-shot  [Hendrycks et al. 2020]
    # ════════════════════════════════════════════════════════════════════
    sep
    log "  [4/7] MMLU  (5-shot, 100 questions across domains)  [Hendrycks et al. 2020]"
    mark_section "MMLU"
    run_mc_section "MMLU" "$MMLU_SYSTEM" MMLU_Q MMLU_ANS
    mmlu_c=$SECTION_CORRECT; mmlu_t=$SECTION_TOTAL; mmlu_s=$SECTION_TIME_S
    mmlu_pct="N/A"
    [ "$mmlu_t" -gt 0 ] && mmlu_pct=$(python3 -c "print(f'{$mmlu_c/$mmlu_t*100:.0f}')")
    R_MMLU["$PICOCLAW_CFG"]="$mmlu_pct"

    if ! assert_alive; then
        stop_resource_sampler 2>/dev/null; stop_heartbeat; oom_at="MMLU"
        R_STATUS["$PICOCLAW_CFG"]="oom@MMLU"
        OOM_MODELS+=("${PICOCLAW_CFG} [OOM at MMLU]")
        MDL_END=$(date +%s); R_TIME["$PICOCLAW_CFG"]=$(( MDL_END - MDL_START ))
        clean_edge; continue
    fi

    # ════════════════════════════════════════════════════════════════════
    #  [5] GSM8K  —  2-shot chain-of-thought  [Cobbe et al. 2021]
    # ════════════════════════════════════════════════════════════════════
    sep
    log "  [5/7] GSM8K  (2-shot CoT, 100 problems)  [Cobbe et al. 2021]"
    mark_section "GSM8K"
    run_gsm8k_section GSM8K_Q GSM8K_ANS
    gsm_c=$SECTION_CORRECT; gsm_t=$SECTION_TOTAL; gsm_s=$SECTION_TIME_S
    gsm_pct="N/A"
    [ "$gsm_t" -gt 0 ] && gsm_pct=$(python3 -c "print(f'{$gsm_c/$gsm_t*100:.0f}')")
    R_GSM8K["$PICOCLAW_CFG"]="$gsm_pct"

    if ! assert_alive; then
        stop_resource_sampler 2>/dev/null; stop_heartbeat; oom_at="GSM8K"
        R_STATUS["$PICOCLAW_CFG"]="oom@GSM8K"
        OOM_MODELS+=("${PICOCLAW_CFG} [OOM at GSM8K]")
        MDL_END=$(date +%s); R_TIME["$PICOCLAW_CFG"]=$(( MDL_END - MDL_START ))
        clean_edge; continue
    fi

    # ════════════════════════════════════════════════════════════════════
    #  [6] TruthfulQA  —  0-shot MC1  [Lin et al. 2021]
    # ════════════════════════════════════════════════════════════════════
    sep
    log "  [6/7] TruthfulQA  (0-shot MC1, 100 questions)  [Lin et al. 2021]"
    mark_section "TruthfulQA"
    run_mc_section "TruthfulQA" "$MC_SYSTEM_0SHOT" TQA_Q TQA_ANS
    tqa_c=$SECTION_CORRECT; tqa_t=$SECTION_TOTAL; tqa_s=$SECTION_TIME_S
    tqa_pct="N/A"
    [ "$tqa_t" -gt 0 ] && tqa_pct=$(python3 -c "print(f'{$tqa_c/$tqa_t*100:.0f}')")
    R_TQA["$PICOCLAW_CFG"]="$tqa_pct"

    if ! assert_alive; then
        stop_resource_sampler 2>/dev/null; stop_heartbeat; oom_at="TruthfulQA"
        R_STATUS["$PICOCLAW_CFG"]="oom@TruthfulQA"
        OOM_MODELS+=("${PICOCLAW_CFG} [OOM at TruthfulQA]")
        MDL_END=$(date +%s); R_TIME["$PICOCLAW_CFG"]=$(( MDL_END - MDL_START ))
        clean_edge; continue
    fi

    # ════════════════════════════════════════════════════════════════════
    #  [7] Long Context / Needle-in-a-Haystack
    #      Kamradt (2023); Liu et al. (2023) arXiv:2307.03172
    # ════════════════════════════════════════════════════════════════════
    sep
    log "  [7/7] Long Context — Needle-in-a-Haystack (3 positions)  [Kamradt 2023; Liu et al. 2023]"
    log "        Needle: 'XK-7749'  |  CTX_SIZE=${CTX_SIZE}  |  KV_cache=${kv_mb} MiB"
    mark_section "LongContext"
    run_long_context_section
    lc_pct="$LC_PCT"; lc_s="$LC_TIME_S"
    R_LONGCTX["$PICOCLAW_CFG"]="$lc_pct"

    if ! assert_alive; then
        stop_resource_sampler 2>/dev/null; stop_heartbeat; oom_at="LongContext"
        R_STATUS["$PICOCLAW_CFG"]="oom@LongCtx"
        OOM_MODELS+=("${PICOCLAW_CFG} [OOM at LongContext]")
        MDL_END=$(date +%s); R_TIME["$PICOCLAW_CFG"]=$(( MDL_END - MDL_START ))
        clean_edge; continue
    fi

    # ── Composite score — average only over sections that completed ───────
    # Sections aborted by unrecoverable OOM report N/A; those are excluded from mean.
    composite=$(python3 -c "
raw = [('ARC-E','${arc_e_pct}'),('ARC-C','${arc_c_pct}'),('HellaSwag','${hs_pct}'),
       ('MMLU','${mmlu_pct}'),('GSM8K','${gsm_pct}'),('TruthfulQA','${tqa_pct}'),('LongCtx','${lc_pct}')]
vals = [(n, float(v)) for n, v in raw if v not in ('N/A', '')]
if vals:
    print(f'{sum(v for _,v in vals)/len(vals):.1f}')
else:
    print('N/A')
")
    R_COMPOSITE["$PICOCLAW_CFG"]="$composite"
    R_STATUS["$PICOCLAW_CFG"]="done"
    DONE_MODELS+=("$PICOCLAW_CFG")

    stop_heartbeat

    # ── Stop resource sampler, log per-section table ──────────────────
    resource_output=$(stop_resource_sampler)
    MDL_END=$(date +%s)
    R_TIME["$PICOCLAW_CFG"]=$(( MDL_END - MDL_START ))

    sep
    log "  RESULTS SUMMARY — ${PICOCLAW_CFG}"
    log "    ARC-Easy     : ${arc_e_c}/${arc_e_t} = ${arc_e_pct}%  ($(fmt_time $arc_e_s))"
    log "    ARC-Challenge: ${arc_c_c}/${arc_c_t} = ${arc_c_pct}%  ($(fmt_time $arc_c_s))"
    log "    HellaSwag    : ${hs_c}/${hs_t} = ${hs_pct}%  ($(fmt_time $hs_s))"
    log "    MMLU(5-shot) : ${mmlu_c}/${mmlu_t} = ${mmlu_pct}%  ($(fmt_time $mmlu_s))"
    log "    GSM8K(CoT)   : ${gsm_c}/${gsm_t} = ${gsm_pct}%  ($(fmt_time $gsm_s))"
    log "    TruthfulQA   : ${tqa_c}/${tqa_t} = ${tqa_pct}%  ($(fmt_time $tqa_s))"
    log "    LongCtx(Ndl) : ${lc_pct}%  (3 positions)  ($(fmt_time $lc_s))"
    log "    KV cache     : ${kv_mb} MiB pre-allocated  (ctx=${CTX_SIZE} tokens)"
    log "    ─────────────────────────────────────────"
    log "    Composite    : ${composite}%  (mean of 7 sections)"
    log "    Total time   : $(fmt_time ${R_TIME[$PICOCLAW_CFG]})"
    log ""
    log "  RESOURCE USAGE (sampled every 2s during inference):"
    log "$(printf '    %-14s │ %8s │ %9s │ %8s │ %9s │ %s' \
        'Section' 'CPUavg%' 'CPUpeak%' 'RAMavg' 'RAMpeak' 'Duration')"
    log "$(printf '    %-14s │ %8s │ %9s │ %8s │ %9s │ %s' \
        '──────────────' '────────' '─────────' '────────' '─────────' '────────')"
    parse_and_log_resources "$resource_output" "$PICOCLAW_CFG"
    sep
    log ""

    CURRENT_SECTION="cleanup"; CURRENT_Q="none"
    track_step "clean_edge"
    stop_server
    ssh "$SSH_TARGET" "rm -f ${REMOTE_MODEL_DIR}/${MODEL_FILE}" 2>/dev/null || true
    sleep 2
    log "  Cleanup done."
    log ""
done

# ═══════════════════════════════════════════════════════════════════════
#  Final results table — sorted by composite score, with baselines
# ═══════════════════════════════════════════════════════════════════════
BENCH_END=$(date +%s)
TOTAL_TIME=$(( BENCH_END - BENCH_START ))

sep2
log " RESULTS — COMPOSITE & PER-SECTION ACCURACY (%)"
log " Benchmark time: $(fmt_time $TOTAL_TIME)"
log " Baselines: Open LLM Leaderboard [6], full-dataset normalized accuracy"
sep2

# Sort completed models by composite score
sorted_entries=()
for mdl in "${!R_COMPOSITE[@]}"; do
    sorted_entries+=("${R_COMPOSITE[$mdl]}|$mdl")
done
IFS=$'\n' sorted=($(printf '%s\n' "${sorted_entries[@]}" | sort -t'|' -k1 -rn))
unset IFS

# Column widths
COL="%-4s %-16s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-10s %-9s %-9s %-9s %-8s %-8s"
HDR1="Rank Model            ARC-E(%) ARC-C(%) HSwag(%) MMLU(%)  GSM8K(%) TruthQA  LongCtx  Composite  BaseCmp  CPUavg%  CPUpeak% RAMpeak  KV_MiB   Time"
HDR2="---- ---------------- -------- -------- -------- -------- -------- -------- -------- ---------  -------  -------  -------- -------  ------   --------"

log ""; log "$HDR1"; log "$HDR2"

rank=0
for entry in "${sorted[@]}"; do
    comp="${entry%%|*}"; mdl="${entry#*|}"
    rank=$(( rank + 1 ))

    arc_e="${R_ARC_E[$mdl]:-N/A}"
    arc_c="${R_ARC_C[$mdl]:-N/A}"
    hs="${R_HELLASWAG[$mdl]:-N/A}"
    mmlu="${R_MMLU[$mdl]:-N/A}"
    gsm="${R_GSM8K[$mdl]:-N/A}"
    tqa="${R_TQA[$mdl]:-N/A}"
    lc="${R_LONGCTX[$mdl]:-N/A}"
    kv="${R_KV_MB[$mdl]:-N/A}"
    status="${R_STATUS[$mdl]:-?}"
    elapsed="${R_TIME[$mdl]:-0}"

    # Compare to published baseline composite (if known)
    # Note: baselines cover 6 sections; composite now includes LongCtx as 7th.
    # BaseCmp is computed against the 6-section baseline for apples-to-apples.
    baseline_cmp="N/A"
    if [ -n "${BASELINES[$mdl]:-}" ]; then
        baseline_cmp=$(python3 - "${BASELINES[$mdl]}" "$comp" << 'PYEOF'
import sys
parts = [float(x) for x in sys.argv[1].split('|')]
# baselines: ARC_E, ARC_C, HellaSwag, MMLU, GSM8K, TruthfulQA (6 vals)
base_composite = sum(parts) / len(parts)
achieved = float(sys.argv[2])
delta = achieved - base_composite
sign = "+" if delta >= 0 else ""
print(f"{sign}{delta:.1f}pp")
PYEOF
)
    fi

    status_flag=""
    [ "$status" != "done" ] && status_flag=" ⚠${status}"

    cpu_avg="${R_CPU_AVG[$mdl]:-N/A}"
    cpu_pk="${R_CPU_PEAK[$mdl]:-N/A}"
    ram_pk="${R_RAM_PEAK[$mdl]:-N/A}"
    [ "$cpu_avg" != "N/A" ] && cpu_avg="${cpu_avg}%"
    [ "$cpu_pk"  != "N/A" ] && cpu_pk="${cpu_pk}%"
    [ "$ram_pk"  != "N/A" ] && ram_pk="${ram_pk}MB"
    [ "$kv"      != "N/A" ] && kv="${kv}Mi"

    log "$(printf "$COL" \
        "#${rank}" \
        "${mdl}${status_flag}" \
        "${arc_e}%" \
        "${arc_c}%" \
        "${hs}%" \
        "${mmlu}%" \
        "${gsm}%" \
        "${tqa}%" \
        "${lc}%" \
        "${comp}%" \
        "${baseline_cmp}" \
        "$cpu_avg" \
        "$cpu_pk" \
        "$ram_pk" \
        "$kv" \
        "$(fmt_time $elapsed)")"
done

# Published baseline row for reference
log ""
log "  Published baselines (Open LLM Leaderboard [6], full dataset, %):"
log "$(printf "$COL" "" "Model" "ARC-E" "ARC-C" "HSwag" "MMLU" "GSM8K" "TruthQA" "LongCtx" "Composite" "Source" "" "" "" "" "")"
log "$(printf "$COL" "" "----------------" "--------" "--------" "--------" "--------" "--------" "--------" "--------" "---------" "-------" "" "" "" "" "")"
for entry in "${sorted[@]}"; do
    mdl="${entry#*|}"
    if [ -n "${BASELINES[$mdl]:-}" ]; then
        IFS='|' read -r b_arc_e b_arc_c b_hs b_mmlu b_gsm b_tqa <<< "${BASELINES[$mdl]}"
        b_comp=$(python3 -c "print(f'{($b_arc_e+$b_arc_c+$b_hs+$b_mmlu+$b_gsm+$b_tqa)/6:.1f}')")
        # Baselines don't include LongCtx — show N/A for that column
        log "$(printf "$COL" "" "${mdl}" "${b_arc_e}%" "${b_arc_c}%" "${b_hs}%" "${b_mmlu}%" "${b_gsm}%" "${b_tqa}%" "N/A" "${b_comp}%" "[6]" "" "" "" "" "")"
    fi
done

log ""
log "  Note: Composite = mean of 7 sections (ARC-E/C, HellaSwag, MMLU, GSM8K, TruthfulQA, LongCtx)."
log "        BaseCmp = achieved composite − published 6-section baseline composite."
log "        Negative BaseCmp = quantization/edge degradation vs full model."
log "        KV_MiB = pre-allocated KV cache from llama-server startup log."

# OOM list
if [ "${#OOM_MODELS[@]}" -gt 0 ]; then
    log ""
    log "  ⚠  CRASH / OOM LIST (${#OOM_MODELS[@]} model(s)):"
    for entry in "${OOM_MODELS[@]}"; do
        log "     • $entry"
    done
    log ""
    log "  Diagnostic: ssh ${SSH_TARGET} 'dmesg | grep -i oom | tail -5'"
else
    log ""
    log "  ✓ No crashes — all models completed all sections."
fi

sep2
log ""
log " Report : ${REPORT_FILE}"
log " Done   : $(date '+%Y-%m-%d %H:%M:%S')"
sep2
