import sys
import unittest
import tempfile
import json
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import torch
from wire import parse, compile, legal_targets, serialize, State
from wire_pointer import DevicePointer
from data import Vocab, BOS, SEP, EOS
from model import Config, build_model
import sample
import data


def vocabulary():
    from data import SPECIAL
    from wire import EXTRA
    from spatial import NUMS, DIRS, ITEM_WORDS, TARGETS
    words = set(LEVEL.split()) | set(NUMS) | set(DIRS) | set(ITEM_WORDS) | set(TARGETS)
    return Vocab(SPECIAL + sorted(words) + EXTRA)


LEVEL = ('size 5 5 5 open 1 1 1 3 3 3 entry 1 1 1 +x exit 3 3 1 -x '
         'button 2 2 1 cube > exit > polarity 0 funnel 2 3 1 +x off rev')


class WireTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def model(self):
        v = vocabulary()
        torch.manual_seed(3)
        return build_model(Config(len(v), 128, 2, 2, 32, 0.0, arch='wire', itos=tuple(v.itos))), v

    def test_structured_roundtrip(self):
        state = State()
        for word in ['<bos>', '<sep>'] + serialize(LEVEL):
            state.feed(word)
        self.assertEqual(state.finish(), LEVEL)

    def test_loader_split_cache_and_away_coordinates(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            vf = root / 'vocab.txt'
            vf.write_text(' '.join(w for w in vocabulary().itos if not w.startswith('<')))
            flat = Vocab.build(vf)
            wired = Vocab.build(vf, arch='wire')
            path = root / 'levels.jsonl'
            row = dict(tokens=LEVEL, verdict='solvable', workshop_id=123,
                       game_items=[dict(name='tbeam', pos=[2, 3, 1])])
            path.write_text(json.dumps(row) + '\n' + json.dumps(row) + '\n')
            with patch.object(data, 'LEVEL_CACHE', root / 'cache'):
                a, b, skipped = data.load([path], flat, 256)
                c, e, skipped_wire = data.load([path], wired, 256)
                self.assertEqual((len(a), len(b), skipped), (len(c), len(e), skipped_wire))
                chosen = c if len(c) else e
                ids, _ = chosen.sequence(0, 0)
                self.assertEqual(data.identity(row), 'ws 123')
                words = wired.decode(ids[ids.index(SEP)+1:-1])
                from wire import deserialize
                self.assertEqual(deserialize(words), LEVEL)
                again = data.load([path], wired, 256)
                self.assertEqual((len(again[0]), len(again[1])), (len(c), len(e)))
                m = build_model(Config(len(wired), 256, 1, 2, 32, 0.0,
                                       arch='wire', itos=tuple(wired.itos)))
                x, y, count = chosen.batch([0, 1], 0, 'cpu', m.featurize)
                _, loss = m(x, y)
                self.assertTrue(torch.isfinite(loss))
                self.assertGreater(count, 0)
                away = data.Away([path], wired, 256)
                ax, ay = away.batch(1, 0, 'cpu', m.featurize)
                self.assertEqual(wired.decode(ay[ay != -100].tolist()), ['2', '3', '1'])
                self.assertTrue(torch.isfinite(m(ax)[0][ay != -100]).all())

    def test_constraint_and_compiler_on_each_prefix(self):
        from wire_model import WireConstraint
        m, v = self.model()
        p = [BOS, SEP]
        seq = v.encode(serialize(LEVEL))
        class AllowAll:
            def __call__(self, indices, seqs):
                return torch.ones((len(indices), len(v)), dtype=torch.bool)
        constraint = WireConstraint(m, [p], AllowAll())
        for at, token in enumerate(seq):
            allowed = constraint([0], [seq[:at]])
            self.assertTrue(allowed[0, token], (at, v.itos[token]))
            self.assertFalse(allowed[0, v.stoi['>']])
        self.assertTrue(constraint([0], [seq])[0, EOS])
        self.assertEqual(' '.join(v.decode(constraint.finish(0, seq))), LEVEL)

    def test_no_future_features_and_left_padding(self):
        m, v = self.model()
        seq = [BOS, SEP] + v.encode(serialize(LEVEL))
        full = m.featurize(torch.tensor([seq]))
        for end in range(1, len(seq)+1):
            torch.testing.assert_close(m.featurize(torch.tensor([seq[:end]])), full[:, :end])
        ids = torch.tensor([[0, 0] + seq])
        positions = (torch.arange(ids.shape[1])-2).clamp(min=0)[None]
        real = ids != 0
        mask = torch.ones(ids.shape[1], ids.shape[1], dtype=torch.bool).tril()[None] & real[:, None]
        mask |= torch.eye(ids.shape[1], dtype=torch.bool)[None]
        with torch.no_grad():
            padded = m(ids, positions=positions, mask=mask[:, None])[0]
            plain = m(torch.tensor([seq]))[0]
        torch.testing.assert_close(padded[:, 2:], plain, atol=1e-5, rtol=1e-4)

    def test_cache_gradients_reload(self):
        m, v = self.model()
        level = LEVEL + ' button 1 2 1 cube > polarity 1 funnel 3 2 1 +x off'
        ids = torch.tensor([[BOS, SEP] + v.encode(serialize(level)) + [EOS]])
        x, y = ids[:, :-1], ids[:, 1:]
        full, loss = m(x, y)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(m.pointer.query.weight.grad.abs().sum().item(), 0)
        feats = m.featurize(x)
        caches = [{'size': x.shape[1]} for _ in m.blocks]
        with torch.no_grad():
            chunks = [m(x[:, :7], caches=caches)[0]]
            for i in range(7, x.shape[1]):
                chunks.append(m((x[:, i:i+1], feats[:, i:i+1]), caches=caches, start=i)[0])
        torch.testing.assert_close(torch.cat(chunks, 1), full, atol=1e-5, rtol=1e-4)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'wire.pt'
            torch.save(dict(model=m.state_dict(), model_config=m.cfg.__dict__, vocab=v.itos), p)
            restored, _, _ = sample.load_checkpoint(p, 'cpu')
            torch.testing.assert_close(restored(x)[0], full)

    def test_failure_accounting_and_refill(self):
        m, v = self.model()
        # A valid body supplied as a prefix makes this exercise actual pointer sampling,
        # and two different lengths exercise prompt-cache replacement and row reuse.
        body = serialize(LEVEL)
        body = body[:body.index('<wires>')]
        p = [BOS, SEP] + v.encode(body)
        short = [BOS, SEP] + v.encode('size 5 5 5 open 1 1 1 3 3 3 entry 1 1 1 +x exit 3 3 1 -x'.split())
        prompts = [p, short, p, short, p]
        together = m.generate_many(prompts, 40, EOS, top_k=1, batch=2)
        alone = [m.generate([prompt], 40, EOS, top_k=1)[0] for prompt in prompts]
        self.assertEqual(together, alone)
        self.assertEqual(m.generate_refill(p, 3, 40, EOS, top_k=1, batch=2), [alone[0]] * 3)
        failed = m.generate_many([[BOS, SEP]] * 5, 1, EOS, top_k=1, batch=2)
        self.assertEqual(failed, [[0]] * 5)
        self.assertEqual(m.last_generation_stats['compile_failures'], 5)

    def test_forward_reference_and_polarity(self):
        records = parse(LEVEL)
        self.assertEqual(compile(records), LEVEL)
        self.assertEqual(records.wires[1].target, 3)
        self.assertEqual(legal_targets(records.devices, 2, 'polarity'),
                         [False, False, False, True])
        devices = [('button', '1', '1', '1', 'cube')] + [('laser',)] * 34
        allowed = legal_targets(devices, 0, 'laser')
        self.assertEqual(sum(allowed), 33)
        self.assertFalse(allowed[-1])

    def test_bad_reference(self):
        with self.assertRaises(ValueError):
            parse(LEVEL.replace('polarity 0', 'polarity 1'))
        from wire import deserialize, pointer_index
        for token in ('2', '<ptr:-1>', '<ptr:256>', '<ptr:2x', '<ptr:02>'):
            with self.subTest(token=token), self.assertRaises(ValueError):
                pointer_index(token)
        with self.assertRaises(ValueError):
            deserialize(serialize(LEVEL)[:-1])
        with self.assertRaises(ValueError):
            parse(LEVEL.replace('polarity 0', 'gel 0') + ' gel 2 2 1 +z bounce splat')

    def test_pointer_masks_and_gradients(self):
        pointer = DevicePointer(8)
        queries = torch.randn(2, 3, 8, requires_grad=True)
        devices = torch.randn(2, 4, 8, requires_grad=True)
        allowed = torch.ones(2, 3, 4, dtype=torch.bool)
        allowed[..., 0] = False
        logits, loss = pointer(queries, devices, allowed, torch.ones(2, 3, dtype=torch.long))
        self.assertTrue(torch.isneginf(logits[..., 0]).all())
        loss.backward()
        self.assertTrue(torch.isfinite(queries.grad).all())
        self.assertTrue(torch.isfinite(devices.grad).all())
        with self.assertRaises(ValueError):
            pointer(queries, devices, torch.zeros_like(allowed))


if __name__ == '__main__':
    unittest.main()
