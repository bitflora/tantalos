"""Loss of a checkpoint on held-out Workshop levels (levels with a token it doesn't know are skipped).

    python train/ws_loss.py data/ws/test.jsonl CHECKPOINT...
"""
import json, sys
import torch
sys.path.insert(0, __import__("os").path.dirname(__file__))
import data, sample

path, cks = sys.argv[1], sys.argv[2:]
recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
dev = "cuda" if torch.cuda.is_available() else "cpu"
for ck in cks:
    model, vocab, _ = sample.load_checkpoint(ck, dev)
    model.eval()
    losses = []
    for r in recs:
        toks = r["tokens"].split()
        if any(t not in vocab.stoi for t in toks) or len(toks) + 40 > model.cfg.block_size:
            continue
        ids = vocab.encode(data.slots([c for c in data.conditions(r) if c in vocab.stoi], data.groups_in(vocab)))
        lv = data.Levels(torch.tensor(vocab.encode(toks), dtype=torch.int16), torch.tensor([0, len(toks)]), [ids], vocab.encode([data.ANY[g] for g in data.groups_in(vocab)]))
        x, y, _ = lv.batch([0], 0, dev)
        with torch.no_grad():
            losses.append(model(x, y)[1].item())
    print(f"{ck}: {len(losses)}/{len(recs)} levels, mean loss {sum(losses)/max(len(losses),1):.4f}")
