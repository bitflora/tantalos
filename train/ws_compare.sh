#!/usr/bin/env bash
# Two 2-hour runs on generated v3r levels + Workshop levels (data/ws; train/ws_loss.py scores held-out ones):
#   runs/ws/warm  = loop-03/best.pt continued; runs/ws/fresh = new net.  bash train/ws_compare.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
DATA="data/v3r/levels-1.jsonl data/v3r/levels-2.jsonl data/ws/train-rep.jsonl data/ingame/invalid.jsonl"
log() { echo "[$(date +%T)] $*"; }
$PY -c "import ctypes, time; ctypes.windll.kernel32.SetThreadExecutionState(0x80000001); time.sleep(5*3600)" &
trap 'kill $! 2>/dev/null || true' EXIT
run() {  # run NAME NOTE args...
    name=$1; note=$2; shift 2
    mkdir -p runs/ws/$name
    log "$name: training"
    $PY -u train/train.py --data $DATA --vocab data/ws/vocab.txt --out runs/ws/$name --block 1024 --batch 16 \
        --until "$(date -d '+2 hours' +'%Y-%m-%d %H:%M:%S')" --eval-every 2000 --score-n 64 --note "$note" "$@" > runs/ws/$name/train.log 2>&1 || log "$name: train failed"
    log "$name: $(grep '^==' runs/ws/$name/train.log | tail -1)"
    $PY -u train/evaluate.py runs/ws/$name/best.pt --data data/v3r/levels-1.jsonl > runs/ws/$name/evaluate.log 2>&1 || log "$name: evaluate failed"
}
run warm "Workshop test: loop-03/best.pt continued 2h on v3r + 94 Workshop solvable levels x20 (+invalid), new vocab" \
    --init runs/v3-night/loop-03/best.pt --lr 2e-4 --min-lr 2e-5 --warmup 200
run fresh "Workshop test: new net from scratch 2h on v3r + 94 Workshop solvable levels x20 (+invalid)" \
    --lr 1e-3 --min-lr 1e-4 --warmup 300
$PY train/ws_loss.py data/ws/test.jsonl runs/v3-night/loop-03/best.pt runs/ws/warm/best.pt runs/ws/fresh/best.pt > runs/ws/ws_loss.txt 2>&1
log "done"
