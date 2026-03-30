#!/usr/bin/env bash
# ── api.sh ────────────────────────────────────────────────────────────
# HTTP calls to the llama-server OpenAI-compatible endpoint and response
# parsers for multiple-choice and numeric answer extraction.

# ── Response state globals ────────────────────────────────────────────
# Set by call_api(); read by benchmark runners.
LAST_RESPONSE=""
LAST_LATENCY_MS=0
LAST_PROMPT_TOKENS=0
LAST_COMPLETION_TOKENS=0

# ── API call ──────────────────────────────────────────────────────────
# Sends a chat-completion request and populates the globals above.
# On curl/network failure LAST_RESPONSE is set to "ERROR".
call_api() {
    local system_prompt="$1"
    local user_prompt="$2"
    local max_tokens="$3"

    local t0; t0=$(ms_ts)
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

    local t1; t1=$(ms_ts)
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
# Returns A/B/C/D or "?" when no letter can be found.
# Priority: explicit "Answer: X" → standalone leading letter → first A-D anywhere.
parse_mc() {
    local response="$1"
    python3 - "$response" << 'PYEOF'
import sys, re
text = sys.argv[1].strip()
for pat in [
    r'[Aa]nswer[:\s]+\(?([A-D])\)?',
    r'[Tt]he answer is[:\s]+\(?([A-D])\)?',
    r'[Cc]orrect answer[:\s]+\(?([A-D])\)?',
    r'^\s*\(?([A-D])\)?[\s\.\)]',
]:
    m = re.search(pat, text, re.MULTILINE)
    if m:
        print(m.group(1).upper()); sys.exit(0)
m = re.search(r'\b([A-D])\b', text)
if m:
    print(m.group(1).upper()); sys.exit(0)
print("?")
PYEOF
}

# ── Numeric answer extractor (GSM8K) ─────────────────────────────────
# Priority: "#### N" format → "answer is N" / "= N" → last integer in response.
parse_number() {
    local response="$1"
    python3 - "$response" << 'PYEOF'
import sys, re
text = sys.argv[1]
m = re.search(r'####\s*([\d,]+\.?\d*)', text)
if m:
    print(m.group(1).replace(',', '').rstrip('.')); sys.exit(0)
for pat in [r'[Aa]nswer[:\s]+\$?([\d,]+)', r'=\s*([\d,]+)\s*\.?\s*$']:
    m = re.search(pat, text, re.MULTILINE)
    if m:
        print(m.group(1).replace(',', '')); sys.exit(0)
nums = re.findall(r'\b(\d+)\b', text)
print(nums[-1] if nums else "?")
PYEOF
}
