"""CPU_MECHANISM_DEMO. No optimizer, model assets, data or network."""
import json
import torch
from nested_sleep import GradientWindow, LinearSelfModifyingReference

def run():
    torch.manual_seed(17)
    p = torch.tensor([10.])
    w = GradientWindow.create(p, 2).accumulate(torch.tensor([2.]), .1)
    restored = GradientWindow.from_state_dict(w.state_dict())
    w = w.accumulate(torch.tensor([3.]), .1)
    restored = restored.accumulate(torch.tensor([3.]), .1)
    torch.testing.assert_close(w.prospective(p), restored.prospective(p), rtol=0, atol=0)
    m = LinearSelfModifyingReference(3)
    _, a, _ = m(torch.ones(3))
    before = {k:v.clone() for k,v in a.items()}
    m(-torch.ones(3))
    assert all(torch.equal(a[k], before[k]) for k in a)
    assert p.item() == 10
    return {'level':'CPU_MECHANISM_DEMO','prospective':w.prospective(p).item(),
            'live':p.item(),'events':w.events,'isolated':True,'optimizer_attempts':0}
if __name__ == '__main__':
    print(json.dumps(run()))
