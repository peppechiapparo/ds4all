#!/usr/bin/env bash
#
# hybrid_regression.sh -- A/B/C regression check for the DS4_HYBRID_MOE spike
# (CPU/GPU hybrid MoE offload, see ds4_hybrid_moe_enabled() in ds4.c and
# ds4_gpu_hybrid_moe_forward_one()/_batch() in ds4_cuda.cu).
#
# Standalone script, NOT wired into `make test`: it needs the real 81 GB
# ds4flash.gguf model and a CUDA GPU, neither of which the official test
# vectors assume. Run manually from the ds4/ directory after a CUDA build:
#
#   ./tests/hybrid_regression.sh
#
# It runs the same short prompt three times:
#   A) plain CPU backend, no hybrid flags at all (upstream behavior)
#   B) --hybrid-moe --hybrid-no-decode (prefill offloaded to GPU, decode on CPU
#      -- the config validated as a net win on this machine)
#   C) --hybrid-moe (prefill AND decode offloaded to GPU)
#
# A and B must produce bit-for-bit identical generated text: the hybrid
# prefill path is expected to be numerically exact, so any difference is a
# regression (FAIL). C's decode path uses GPU fast-math kernels and is known
# to drift from the CPU path after enough tokens, so a difference there is
# only reported as a WARN, not a FAIL.
set -u

cd "$(dirname "$0")/.." || exit 1

DS4_MODEL="${DS4_MODEL:-./ds4flash.gguf}"
DS4_BIN="./ds4"
PROMPT="Oggi il tempo è sereno e il cielo è azzurro sopra la città. Oggi il tempo è sereno e il cielo è azzurro sopra la città. Oggi il tempo è sereno e il cielo è azzurro sopra la città."
COMMON_ARGS=(--cpu -m "$DS4_MODEL" --ctx 1024 --nothink --temp 0 --tokens 12 -p "$PROMPT")
RUN_TIMEOUT=900

if [[ ! -x "$DS4_BIN" ]]; then
    echo "FAIL: $DS4_BIN not found or not executable (run make first)"
    exit 1
fi

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

# Clear every hybrid env var so each run only depends on its own CLI flags,
# regardless of what the caller's shell has exported.
clear_hybrid_env() {
    unset DS4_HYBRID_MOE DS4_HYBRID_PREFILL DS4_HYBRID_DECODE \
          DS4_HYBRID_T2_GB DS4_HYBRID_MOE_PROFILE
}

# extract_text FILE -- strips ds4's own log/progress lines from a captured
# run, leaving only the generated text. When stderr is redirected to a file
# the progress writer sometimes lands mid-line, gluing a stray character in
# front of the next "ds4: ..." log line (e.g. "Mids4: DS4_HYBRID_DECODE=0
# ..."). A plain `grep -v 'ds4:'` would drop the whole line, including the
# glued-on generated text ("Mi"). Instead, first force every "ds4:"
# occurrence onto its own line with `sed`, then filter with an anchored
# `grep -v '^ds4:'` so only ds4's own lines are dropped and any text glued
# in front of them survives.
extract_text() {
    tr '\r' '\n' < "$1" | sed 's/ds4:/\nds4:/g' | grep -v '^ds4:' | sed '/^[[:space:]]*$/d'
}

run_case() {
    local label="$1" outfile="$2"
    shift 2
    clear_hybrid_env
    timeout "$RUN_TIMEOUT" "$DS4_BIN" "${COMMON_ARGS[@]}" "$@" > "$outfile" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "FAIL: run $label exited with status $rc (see $outfile)"
        cat "$outfile"
        exit 1
    fi
}

echo "== hybrid_regression: model=$DS4_MODEL =="

echo "-- running A (CPU baseline, no hybrid flags) --"
run_case A "$TMPDIR/a.log"

echo "-- running B (--hybrid-moe --hybrid-no-decode) --"
run_case B "$TMPDIR/b.log" --hybrid-moe --hybrid-no-decode

echo "-- running C (--hybrid-moe, full) --"
run_case C "$TMPDIR/c.log" --hybrid-moe

extract_text "$TMPDIR/a.log" > "$TMPDIR/a.txt"
extract_text "$TMPDIR/b.log" > "$TMPDIR/b.txt"
extract_text "$TMPDIR/c.log" > "$TMPDIR/c.txt"

status=0

if diff -q "$TMPDIR/a.txt" "$TMPDIR/b.txt" > /dev/null; then
    echo "PASS: A == B (CPU baseline matches --hybrid-moe --hybrid-no-decode)"
else
    echo "FAIL: A != B -- hybrid prefill offload changed the generated text"
    diff -u "$TMPDIR/a.txt" "$TMPDIR/b.txt"
    status=1
fi

if diff -q "$TMPDIR/a.txt" "$TMPDIR/c.txt" > /dev/null; then
    echo "PASS: A == C (--hybrid-moe full matches CPU baseline)"
else
    echo "WARN: A != C -- known GPU fast-math drift on the decode path, not a fail"
    diff -u "$TMPDIR/a.txt" "$TMPDIR/c.txt"
fi

exit $status
