#!/usr/bin/env bash
# One unattended night: let the running dataset job generate on its own (CPU and GPU share
# the laptop's cooling, so running both slows both), then train from the first run's
# weights on everything until morning, then evaluate.
#
#   bash train/night.sh [GEN_END=22:45] [TRAIN_END=07:40]
#
# Expects `glados-data dataset ... --out data/night/levels.jsonl` already running, and
# runs/first finishing (or finished).
set -euo pipefail
cd "$(dirname "$0")/.."
GEN_END=${1:-22:45}
TRAIN_END=${2:-07:40}
PY=.venv/Scripts/python
mkdir -p runs/night

log() { echo "[$(date +%T)] $*"; }

# Windows sleeps when idle; ask it not to while this script runs.
$PY -c "import ctypes, time; ctypes.windll.kernel32.SetThreadExecutionState(0x80000001); time.sleep(12 * 3600)" &
KEEPER=$!
trap 'kill $KEEPER 2>/dev/null || true' EXIT

log "waiting for runs/first to reach step 20000"
until grep -q "^== step 20000:" runs/first.log 2>/dev/null; do sleep 30; done
log "runs/first done"

log "generating until $GEN_END"
gen_end=$(date -d "$GEN_END" +%s)
while [ "$(date +%s)" -lt "$gen_end" ]; do sleep 20; done
taskkill //F //IM glados-data.exe > /dev/null 2>&1 || true
sleep 2
# Keep whole lines only; the generator was stopped mid-write.
n=$(wc -l < data/night/levels.jsonl)
head -n "$n" data/night/levels.jsonl > data/night/part.jsonl
log "kept $n new levels in data/night/part.jsonl"

log "training until $TRAIN_END"
$PY -u train/train.py --data data/levels.jsonl data/night/part.jsonl --out runs/night \
    --init runs/first/best.pt --block 640 --until "$TRAIN_END" \
    --lr 6e-4 --min-lr 3e-5 --warmup 500 --eval-every 2000 --score-n 64 \
    > runs/night/train.log 2>&1
log "training finished: $(grep '^==' runs/night/train.log | tail -1)"

log "evaluating runs/night/best.pt"
$PY -u train/evaluate.py runs/night/best.pt --data data/levels.jsonl data/night/part.jsonl \
    > runs/night/evaluate.log 2>&1
log "evaluation written to runs/night/eval/report.md"

log "checking samples in Portal 2's editor"
$PY -u train/ingame.py runs/night/eval > runs/night/ingame.log 2>&1 \
    || log "in-game check failed; see runs/night/ingame.log"
