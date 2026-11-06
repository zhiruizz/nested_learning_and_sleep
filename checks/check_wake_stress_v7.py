"""Explicit CPU eight-tick control check; separate from the ordinary update budget."""
from pathlib import Path
import importlib.util,sys,copy
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('tail_fixture',ROOT/'tests/test_agnews_tail_cache_v7_cpu.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
from stress_agnews_single_receiver_v7 import wake_step
from nested_sleep.optim import CMSController
from matmul_work_v7 import MatmulWork


class Toy(fixture.TinyLlamaCMSLM):
    def banks(self):return [level.experts for level in self.backbone.model.layers[-1].mlp.memory]
    def level_descriptors(self):
        prefix='backbone.model.layers.0.mlp.memory'
        return [{'bank':f'{prefix}.{i}.experts','base':[f'{prefix}.{i}.a',f'{prefix}.{i}.b']} for i in (0,1)]


def main():
    torch.manual_seed(763)
    original=Toy().eval();original.banks()[1].activate(0);original.banks()[1].activate(1)
    original.banks()[1].route_override={'mix':1,'fraction':.5}
    features={i:{'y':torch.randn(1,2,6),'residual':torch.randn(1,2,6),'lengths':torch.tensor([2])} for i in range(8)}
    train=[{'class_id':2 if i<4 else 3} for i in range(8)]
    pack={'code_maps':{'17':{'marker_token_id':9,'class_to_token_ids':[10,11,12,13]}}}
    states=[];meter=MatmulWork()
    for writes in (True,False):
        m=copy.deepcopy(original);controller=CMSController(m,(4,16),base_lr=.02,expert_lr=.04)
        before={n:v.clone() for n,v in m.state_dict().items()}
        sequence=[]
        with meter:
            for _ in range(4):
                wake_step(m,controller,features,train,list(range(8)),pack,17,writes,meter)
                sequence.append({n:p.detach().clone() for n,p in m.named_parameters()})
        for name,value in before.items():
            if '.memory.1.experts.' in name:torch.testing.assert_close(value,m.state_dict()[name],rtol=0,atol=0)
        if not writes:
            for d in m.level_descriptors():
                for n in d['base']:torch.testing.assert_close(before[n],m.state_dict()[n],rtol=0,atol=0)
        states.append(sequence)
        assert controller.counts==[0,4]
    for step in range(3):
        for n in states[0][step]:torch.testing.assert_close(states[0][step][n],states[1][step][n],rtol=0,atol=0)
    assert any(not torch.equal(states[0][3][n],states[1][3][n]) for n in original.level_descriptors()[0]['base'])
    print('8 CPU CMS ticks: pre-boundary branches exact; applying/skipping base write diverges at4; frozen slow weights unchanged.')

if __name__=='__main__':main()
