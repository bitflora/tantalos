#!/usr/bin/env bash
# Train/validate loops until a clock time. Each loop warm-starts from the previous loop's final
# weights and runs retrain.sh (train, evaluate, check the samples in Portal 2, train away from
# every level the game has rejected, evaluate and check again), so the levels the game rejects
# in one loop are trained away from in the same loop and the next.
#
#   bash train/loop.sh OUT END INIT [train.py args...]
#   bash train/loop.sh runs/overnight 09:00 runs/v3b/best.pt --lr 1e-4 --min-lr 1e-5 --warmup 100
#
# MINUTES (default 38) is each loop's main training time. Two evaluations and in-game checks add
# about 40 more, plus AWAY_MINUTES (default 15) of training away. A loop only starts if it can
# finish before END. DATA, PY, INGAME, AWAY, AWAY_MINUTES and AWAY_REPEAT pass through to
# retrain.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=$1
END=$2
INIT=$3
shift 3
MINUTES=${MINUTES:-38}
AWAY_MINUTES=${AWAY_MINUTES:-15}
export AWAY_MINUTES
OVERHEAD=$(( 40 + AWAY_MINUTES ))
PY=${PY:-.venv/Scripts/python}
log() { echo "[$(date +%T)] $*"; }

end=$(date -d "$END" +%s)
if [ "$end" -le "$(date +%s)" ]; then end=$(date -d "tomorrow $END" +%s); fi
log "looping until $(date -d "@$end" +"%Y-%m-%d %H:%M"), $MINUTES min of training per loop"

# Windows sleeps when idle; ask it not to between training phases too.
$PY -c "import ctypes, time; ctypes.windll.kernel32.SetThreadExecutionState(0x80000001); time.sleep($(( end - $(date +%s) + 600 )))" &
KEEPER=$!
trap 'kill $KEEPER 2>/dev/null || true' EXIT

mkdir -p "$OUT"
init=$INIT
n=0
while [ $(( $(date +%s) + (MINUTES + OVERHEAD) * 60 )) -le "$end" ]; do
    n=$((n + 1))
    dir=$(printf "%s/loop-%02d" "$OUT" "$n")
    log "loop $n: $dir from $init"
    if bash train/retrain.sh "$dir" "+$MINUTES minutes" --init "$init" "$@" > "$dir.log" 2>&1; then
        log "loop $n: $(grep 'training finished' "$dir.log" | sed 's/.*finished: //')"
        log "loop $n: validity $(grep -A4 '^## Validity' "$dir/eval/report.md" | tail -1)"
        grep 'eval: solvable' "$dir.log" | sed "s/^\[[0-9:]*\] /[$(date +%T)] loop $n: /" || true
    else
        log "loop $n failed; see $dir.log"
    fi
    if [ -f "$dir/away/best.pt" ] && [ ! -f "$dir/away/WORSE" ]; then
        init=$dir/away/best.pt
    elif [ -f "$dir/best.pt" ]; then
        init=$dir/best.pt
    fi
done
log "done after $n loops"
