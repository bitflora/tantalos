#!/usr/bin/env bash
# Train on a generated dataset until a clock time, then evaluate.
#
#   bash train/retrain.sh OUT UNTIL [train.py args...]
#   bash train/retrain.sh runs/v2 07:40
#   bash train/retrain.sh runs/v2 "+3 hours"            # counted from when training starts
#   RELABEL=1 bash train/retrain.sh runs/v2 07:40      # after changing the solver
#   DATA=data/v2 bash train/retrain.sh runs/v2 07:40 --init runs/night/best.pt
#
# DATA (default data/v2) holds `glados dataset` output: levels-*.jsonl and vocab.txt.
# RELABEL=1 first re-scores every level with the current `glados score` into
# $DATA/relabelled/ and trains on that, so solver changes don't need the slow
# generator run again. (The generator's own choices still followed the old solver.)
#
# Levels Portal 2's editor rejected (data/ingame/invalid.jsonl, from earlier runs) are trained
# on as invalid, never relabelled. After evaluating, train/ingame.py checks the samples in the
# game, unless INGAME=0; if the game can't be used (say you're playing it) that's logged and
# skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=$1
UNTIL=$2
shift 2
DATA=${DATA:-data/v2}
PY=.venv/Scripts/python
GLADOS=target/release/glados.exe
log() { echo "[$(date +%T)] $*"; }

cargo build --release --quiet
files=("$DATA"/levels-*.jsonl)
if [ "${RELABEL:-0}" = 1 ]; then
    mkdir -p "$DATA/relabelled"
    relabelled=()
    for f in "${files[@]}"; do
        dest="$DATA/relabelled/$(basename "$f")"
        log "relabelling $f"
        $PY -c "import json, sys; [print(json.loads(l)['tokens']) for l in sys.stdin]" < "$f" \
            | $GLADOS score - > "$dest"
        relabelled+=("$dest")
    done
    files=("${relabelled[@]}")
fi

train_files=("${files[@]}")
if [ -s data/ingame/invalid.jsonl ]; then
    train_files+=(data/ingame/invalid.jsonl)
fi

case $UNTIL in
    +*) UNTIL=$(date -d "$UNTIL" +"%Y-%m-%d %H:%M:%S") ;;
esac
mkdir -p "$OUT"
log "training on ${train_files[*]} until $UNTIL"
$PY -u train/train.py --data "${train_files[@]}" --vocab "$DATA/vocab.txt" --out "$OUT" \
    --block 640 --until "$UNTIL" --eval-every 2000 --score-n 64 "$@" > "$OUT/train.log" 2>&1
log "training finished: $(grep '^==' "$OUT/train.log" | tail -1)"

log "evaluating $OUT/best.pt"
$PY -u train/evaluate.py "$OUT/best.pt" --data "${files[@]}" > "$OUT/evaluate.log" 2>&1
log "evaluation written to $OUT/eval/report.md"

if [ "${INGAME:-1}" = 1 ]; then
    log "checking samples in Portal 2's editor"
    if $PY -u train/ingame.py "$OUT/eval" > "$OUT/ingame.log" 2>&1; then
        log "in-game check: $(grep -m1 'Game-valid, of levels\|Skipped' "$OUT/eval/ingame/report.md")"
    else
        log "in-game check failed; see $OUT/ingame.log"
    fi
fi
