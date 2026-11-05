import io

import torch

from nested_sleep.lifecycle import SleepSession, fingerprint
from nested_sleep.lm import NativeResearchLM
from nested_sleep.native import NativeConfig


def _optimizer_state_signature(state):
    entries = []
    for key, values in sorted(state.items(), key=lambda item: str(item[0])):
        row = {name: (value.detach().clone() if isinstance(value, torch.Tensor) else value)
               for name, value in values.items()}
        entries.append((key, row))
    return entries


def test_sleep_restore_does_not_mutate_checkpoint_optimizer_or_sibling_restore(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(73)
    config = NativeConfig(vocab_size=17, dim=4, hidden=5, layers=1,
                          projection_chunk=4, memory_chunk=1, cms_periods=(1, 2))
    live = NativeResearchLM(config, rank=2, capacity=3).cpu().eval().requires_grad_(False)
    sender = live.level_descriptors()[0]
    params = dict(live.named_parameters())
    prospective = {name: params[name].detach().clone() for name in sender['base']}
    session = SleepSession(live, 0, prospective, seed=19, lr=.01, dream_tokens=2)
    prefix = torch.tensor([[1, 3, 5]], dtype=torch.long)
    session.seed_step(prefix)

    buffer = io.BytesIO()
    torch.save(session.state_dict(), buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, map_location='cpu', weights_only=False)
    checkpoint_optimizer = _optimizer_state_signature(checkpoint['optimizer']['state'])

    first = SleepSession.restore(live, checkpoint)
    second = SleepSession.restore(live, checkpoint)
    for branch in (first, second):
        for saved_id, parameter in zip(checkpoint['optimizer']['param_groups'][0]['params'],
                                      branch.optimizer.param_groups[0]['params']):
            saved_step = checkpoint['optimizer']['state'][saved_id]['step']
            restored_step = branch.optimizer.state[parameter]['step']
            assert saved_step.data_ptr() != restored_step.data_ptr()
    first.seed_step(prefix)
    after_first = _optimizer_state_signature(checkpoint['optimizer']['state'])
    assert all(torch.equal(a[1][k], b[1][k]) if isinstance(a[1][k], torch.Tensor)
               else a[1][k] == b[1][k]
               for a, b in zip(checkpoint_optimizer, after_first)
               for k in a[1])
    assert first.optimizer.state[first.new_slot.a]['step'].item() == 2
    assert second.optimizer.state[second.new_slot.a]['step'].item() == 1

    second.seed_step(prefix)
    assert fingerprint(first.student) == fingerprint(second.student)
