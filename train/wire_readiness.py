"""Reproducible wire-arm readiness gates; no production training run is started.

Use --audit for all default input rows; --smoke for the 32-level overfit, checkpoint,
scorer and full-size memory checks. Reports include all attempted samples.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

import sample
from data import Vocab, BOS, SEP, EOS, GROUPS
from model import Config, build_model
from wire import parse, serialize, deserialize


def audit(root):
    report = {'rows': 0, 'failures': 0, 'files': [], 'max_devices': 0,
              'flat_over_block': 0, 'wire_over_block': 0, 'errors': []}
    prefix = 2 + len(GROUPS) + 1
    for path in sorted((root / 'data/v5b').glob('levels-*.jsonl')):
        file = {'path': str(path), 'rows': 0, 'failures': 0}
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for line in source:
                digest.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                words = row['tokens'].split()
                file['rows'] += 1
                report['rows'] += 1
                try:
                    structured = serialize(words)
                    if deserialize(structured) != ' '.join(words):
                        raise ValueError('roundtrip changed level tokens')
                    report['max_devices'] = max(report['max_devices'], len(parse(words).devices))
                    report['flat_over_block'] += len(words) + prefix > 1024
                    report['wire_over_block'] += len(structured) + prefix > 1024
                except (ValueError, IndexError) as e:
                    file['failures'] += 1
                    report['failures'] += 1
                    if len(report['errors']) < 10:
                        report['errors'].append(dict(path=str(path), row=file['rows'], error=str(e)))
        file['sha256'] = digest.hexdigest()
        report['files'].append(file)
        print(json.dumps(file), flush=True)
    if not report['rows']:
        raise ValueError('no training rows found')
    return report


def smoke(root, scorer, out):
    torch.manual_seed(17)
    torch.set_num_threads(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab = Vocab.build(root / 'data/v5b/vocab.txt', arch='wire')
    levels = []
    seen = set()
    # Actual default training levels, distinct and short, including wired examples.
    for path in sorted((root / 'data/v5b').glob('levels-*.jsonl')):
        with path.open(encoding='utf-8') as source:
            for line in source:
                s = json.loads(line)['tokens']
                if s in seen or ' > ' not in s or len(serialize(s)) > 150:
                    continue
                levels.append(s)
                seen.add(s)
                if len(levels) == 32:
                    break
        if len(levels) == 32:
            break
    if len(levels) != 32:
        raise ValueError('need 32 distinct wired levels')
    (out / 'training-levels.jsonl').write_text(''.join(json.dumps(dict(tokens=s, verdict='uncertain')) + '\n'
                                                     for s in levels), encoding='utf-8')
    prompt = sample.prompt(vocab, [], [])
    seqs = [prompt + vocab.encode(serialize(s)) + [EOS] for s in levels]
    width = max(map(len, seqs)) - 1
    x = torch.zeros((32, width), dtype=torch.long)
    y = torch.full_like(x, -100)
    for r, seq in enumerate(seqs):
        x[r, :len(seq)-1] = torch.tensor(seq[:-1])
        y[r, len(prompt)-1:len(seq)-1] = torch.tensor(seq[len(prompt):])
    model = build_model(Config(len(vocab), 256, 2, 4, 128, 0.0, qk_norm=True,
                               arch='wire', itos=tuple(vocab.itos))).to(device)
    feats = model.featurize(x).to(device)
    x, y = x.to(device), y.to(device)
    opt = model.optimizer(3e-3, 0.0)
    started = time.monotonic()
    for step in range(1500):
        logits, loss = model((x, feats), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            print(f'overfit step={step} loss={loss.item():.4f}', flush=True)
        if step >= 300 and loss.item() < 0.12:
            break
    report = dict(levels=32, loss=loss.item(), steps=step+1, seconds=time.monotonic()-started,
                  device=device, passed_overfit=loss.item() < 0.2)
    model.eval()
    path = out / 'smoke.pt'
    torch.save(dict(model=model.state_dict(), model_config=model.cfg.__dict__, vocab=vocab.itos,
                    step=step+1, val_loss=loss.item()), path)
    restored, _, _ = sample.load_checkpoint(path, device)
    with torch.no_grad():
        torch.testing.assert_close(restored((x, feats))[0], model((x, feats))[0])
    report['checkpoint_reload'] = True
    lines = sample.generate(restored, vocab, 32, temperature=0.5, batch=8,
                            constrained=True, tantalos=scorer)
    results = sample.score(lines, scorer, budget=1000)
    report['generation'] = restored.last_generation_stats
    report['scored'] = len(results)
    report['parsed'] = sum(r['verdict'] != 'unparsed' for r in results)
    (out / 'samples.jsonl').write_text(''.join(json.dumps(dict(sample=s, **r)) + '\n'
                                             for s, r in zip(lines, results)), encoding='utf-8')
    # Separately verify deterministic completions; stochastic truncation is measured above,
    # not retried or removed from its denominator.
    greedy = sample.generate(restored, vocab, 32, top_k=1, batch=8, constrained=True, tantalos=scorer)
    checked = sample.score(greedy, scorer, budget=1000)
    report['greedy_generation'] = restored.last_generation_stats
    report['greedy_parsed'] = sum(r['verdict'] != 'unparsed' for r in checked)
    # Bounded memory smoke on the production-size architecture, one optimizer step.
    del model, restored, opt, logits, loss, x, y, feats
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = build_model(Config(len(vocab), 1024, 6, 6, 384, 0.0, qk_norm=True,
                                   arch='wire', itos=tuple(vocab.itos))).cuda()
        opt = model.optimizer(1e-3, 0.01)
        report['parameters'] = model.n_params()
        # Every token participates in attention, at the production block and batch size.
        # Include dense real endpoint decisions, not just the token decoder's memory.
        base = seqs[0][:-1]
        first_wire = base.index(vocab.stoi['<wire>'])
        links = base[first_wire:]
        long = (base[:first_wire] + links * 1024)[:1025]
        xx = torch.tensor([long[:-1]] * 16, device=device)
        yy = torch.tensor([long[1:]] * 16, device=device)
        ff = model.featurize(xx.cpu()).to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, loss = model((xx, ff), yy)
        if not torch.isfinite(loss):
            raise ValueError('non-finite production-size loss')
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        report['peak_gpu_bytes'] = torch.cuda.max_memory_allocated()
        report['memory_batch'] = 16
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--scorer', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--audit', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {}
    if args.audit:
        report['audit'] = audit(args.data_root)
    if args.smoke:
        if not args.scorer:
            ap.error('--smoke needs --scorer')
        report['smoke'] = smoke(args.data_root, args.scorer, args.out)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    if report.get('audit', {}).get('failures'):
        raise SystemExit(1)
    if args.smoke and (not report['smoke']['passed_overfit'] or report['smoke']['greedy_parsed'] != 32):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
