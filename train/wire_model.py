"""Sampling constraint and compilation adapter for model.WireGPT."""
import torch

from wire import State, TARGETS, SENDERS, legal_targets


class WireConstraint:
    """Translate the standard grammar to the unwired body; mask typed wire records.

    Every completion, including truncated or malformed output, occupies one result slot.
    Compilation failures become <pad>, which the standard scorer counts as unparsed.
    """
    def __init__(self, model, prompts, external):
        self.model, self.prompts, self.external = model, prompts, external
        self.itos, self.stoi = model.cfg.itos, model.stoi
        self.states = []
        self.lengths = [0] * len(prompts)
        self.stats = dict(attempts=len(prompts), compile_failures=0, wires=0, failure_reasons={})
        for prompt in prompts:
            state = State()
            for i in prompt:
                state.feed(self.itos[i])
            self.states.append(state)

    def update(self, i, seq):
        state = self.states[i]
        for token in seq[self.lengths[i]:]:
            state.feed(self.itos[token])
        self.lengths[i] = len(seq)
        return state

    def __call__(self, indices, seqs):
        mask = torch.zeros((len(indices), len(self.itos)), dtype=torch.bool)
        states = [self.update(i, seq) for i, seq in zip(indices, seqs)]
        body_rows = [r for r, s in enumerate(states) if s.phase == 'body']
        body_masks = None
        if self.external is not None and body_rows:
            # Until <wires>, internal tokens are exactly the standard level prefix.
            body_masks = self.external([indices[r] for r in body_rows], [seqs[r] for r in body_rows])
        body_index = 0
        for r, state in enumerate(states):
            allowed = []
            if state.phase == 'body':
                if body_masks is not None:
                    mask[r] = body_masks[body_index]
                    body_index += 1
                    mask[r, self.stoi['<wires>']] = mask[r, self.stoi['<eos>']]
                else:
                    mask[r] = True
                    for word, i in self.stoi.items():
                        if word.startswith('<'):
                            mask[r, i] = False
                    mask[r, self.stoi['<wires>']] = True
                mask[r, self.stoi['<eos>']] = False
                mask[r, self.stoi['>']] = False
            elif state.phase == 'wire':
                allowed = ['<eos>']
                if any(d[0] in SENDERS and any(any(legal_targets(state.records.devices, i, k))
                           for k in TARGETS) for i, d in enumerate(state.records.devices)):
                    allowed.append('<wire>')
            elif state.phase == 'source':
                allowed = [f'<ptr:{i}>' for i, d in enumerate(state.records.devices)
                           if d[0] in SENDERS and any(any(legal_targets(state.records.devices, i, k)) for k in TARGETS)]
            elif state.phase == 'kind':
                allowed = [k for k in TARGETS if any(legal_targets(state.records.devices, state.source, k))]
            elif state.phase == 'target':
                allowed = [f'<ptr:{i}>' for i, ok in enumerate(
                    legal_targets(state.records.devices, state.source, state.kind)) if ok]
            else:
                allowed = ['<eos>']
            for word in allowed:
                mask[r, self.stoi[word]] = True
            if not mask[r].any():
                mask[r, self.stoi['<eos>']] = True
        return mask

    def finish(self, i, seq):
        state = self.update(i, seq)
        try:
            words = state.finish().split()
            # Sampling returns only the continuation, as the standard interface promises.
            prompt = self.prompts[i]
            prefix = prompt[prompt.index(self.stoi['<sep>']) + 1:]
            ids = [self.stoi[w] for w in words]
            if ids[:len(prefix)] != prefix:
                raise ValueError('compiled output changed the supplied prefix')
            self.stats['wires'] += len(state.wires)
            return ids[len(prefix):]
        except (ValueError, KeyError) as e:
            self.stats['compile_failures'] += 1
            reason = str(e)
            self.stats['failure_reasons'][reason] = self.stats['failure_reasons'].get(reason, 0) + 1
            return [self.stoi['<pad>']]
