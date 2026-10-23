"""CPU_MECHANISM_DEMO: fixed position routing, no cross-position mixing."""
import json
from dataclasses import asdict
import torch
from nested_sleep.experts import SparseExperts
from nested_sleep.repairability import Assumptions, classify_position, UNREACHABLE, REACHABLE, UNKNOWN

def run():
    torch.manual_seed(31)
    bank = SparseExperts(2,rank=1,capacity=2).double()
    bank.activate(0);bank.activate(1)
    with torch.no_grad():
        bank.slots[0].route.copy_(torch.tensor([1.,0.]))
        bank.slots[1].route.copy_(torch.tensor([-1.,0.]))
        for slot in bank.slots:
            slot.priority.zero_();slot.a.copy_(torch.tensor([[.2,.3]]));slot.b.zero_()
    x=torch.tensor([[1.,1.],[-1.,1.]],dtype=torch.double)
    before=bank(x).detach().clone()
    # Position1 executes the receiver, yet its A gradient is zero because B=0.
    bank(x)[1].sum().backward()
    assert bank.slots[1].a.grad is not None and bank.slots[1].a.grad.abs().sum()==0
    assert bank.slots[1].b.grad.abs().sum()>0
    with torch.no_grad():
        bank.slots[1].a.add_(.7);bank.slots[1].b.add_(.3)
    after=bank(x).detach()
    assert torch.equal(before[0],after[0]) and not torch.equal(before[1],after[1])
    premises=Assumptions(**{k:True for k in asdict(Assumptions())})
    assert classify_position(False,premises)['status']==UNREACHABLE
    assert classify_position(True,premises,local_gradient_norm=0.)['status']==REACHABLE
    changed=dict(asdict(premises),fixed_tokens=False)
    assert classify_position(False,changed)['status']==UNKNOWN
    return {'level':'CPU_MECHANISM_DEMO','invariant_position_count':1,'positions':2,
            'reachable_zero_A_gradient':True,'changed_tokens_status':UNKNOWN,'optimizer_attempts':0}
if __name__ == '__main__':
    print(json.dumps(run()))
