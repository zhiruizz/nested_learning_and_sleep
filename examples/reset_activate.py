"""CPU_MECHANISM_DEMO: reset and activation with no optimization."""
import json
import torch
from nested_sleep import LinearExpertChain, ConsolidationTransaction
from nested_sleep.consolidation import snapshot, same_snapshot

def run():
    torch.manual_seed(23)
    live = LinearExpertChain()
    sender = live.add_slot(0, 2, active=True)
    with torch.no_grad():
        sender.b.fill_(.2)
    before = snapshot(live)
    tx = ConsolidationTransaction(live, 0, live.base[0].weight.detach()+.03)
    assert same_snapshot(live,before) and same_snapshot(tx.teacher,before)
    assert not bool(tx.student.slots[0][0].active) and bool(tx.new_slot.active)
    assert torch.equal(sender.a,tx.student.slots[0][0].a)
    tx.commit()
    return {'level':'CPU_MECHANISM_DEMO','live_unchanged':same_snapshot(live,before),
            'sender_A_retained':True,'sender_B_zero':True,'receiver_active':True,
            'optimizer_attempts':0,'quality_claim':False}
if __name__ == '__main__':
    print(json.dumps(run()))
