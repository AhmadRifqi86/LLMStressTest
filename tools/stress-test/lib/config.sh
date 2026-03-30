#!/usr/bin/env bash
# ── config.sh ─────────────────────────────────────────────────────────
# Global configuration for the exponential OOM stress test.
# All tunable parameters live here — do not hard-code values elsewhere.

# ── SSH / Edge device ─────────────────────────────────────────────────
SSH_USER="mitlab"
SSH_HOST="192.168.50.233"
SSH_TARGET="${SSH_USER}@${SSH_HOST}"
REMOTE_DIR="~/projects/llamacpp"
REMOTE_MODEL_DIR="${REMOTE_DIR}/model"
LOCAL_MODEL_DIR="./llm"

# ── llama-server ──────────────────────────────────────────────────────
SERVER_PORT=8080
THREADS=4
SERVER_READY_TIMEOUT=120

# ── Inference token budgets ───────────────────────────────────────────
N_PREDICT_MC=64        # Multiple-choice: one letter
N_PREDICT_MATH=512     # GSM8K: chain-of-thought
N_PREDICT_LC=128       # Long-context needle: short retrieval answer

# ── Exponential escalation ────────────────────────────────────────────
# EXPONENTIAL_ROUNDS: doubling rounds per context level.
# Question counts per round r (1-based):
#   min(N, ceil(N / 2^(R-1)) * 2^(r-1))
# Example — N=100, R=6: 4 → 8 → 16 → 32 → 64 → 100
EXPONENTIAL_ROUNDS=6

# CTX_ESCALATION: context window sizes applied in sequence.
# After all EXPONENTIAL_ROUNDS complete at one level, the server is
# restarted with the next context size and rounds begin again.
# Larger ctx → bigger KV cache → more RAM pressure → eventual OOM.
CTX_ESCALATION=(512 1024 2048 4096 8192)

# ── OOM mode: do NOT use RAM-threshold restarts ───────────────────────
# In standard_stress_test.sh this guards against OOM; here we want OOM.
RAM_RESTART_THRESHOLD_MB=0

# ── Model list  (HF_REPO|GGUF_FILE|CONFIG_NAME) ───────────────────────
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

# ── System prompts ────────────────────────────────────────────────────

MC_SYSTEM_0SHOT="You are a knowledgeable assistant. Read the question carefully and answer with only the letter of the correct choice: A, B, C, or D. Do not explain."

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

LC_SYSTEM="You are a precise reading-comprehension assistant. Read the provided text carefully and answer questions based solely on the information it contains."
