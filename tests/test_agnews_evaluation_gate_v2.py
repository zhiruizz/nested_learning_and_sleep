"""CPU-only validation of the real post-seal evaluation gate."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1]/'tools'/'evaluate_agnews_retention_v2.py'
if not SOURCE.exists():
    SOURCE = Path(__file__).with_name('evaluate_agnews_retention_v2.py')
spec = importlib.util.spec_from_file_location('evaluation', SOURCE)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class EvaluationSealGate(unittest.TestCase):
    def fixture(self, root):
        cp = root/'dummy.pt'
        cp.write_bytes(b'CPU-fixture-only')
        digest = evaluation.sha(cp)
        protocol = {'status':'training_sealed','completed_runs':15}
        (root/'training-protocol.json').write_text(json.dumps(protocol))
        foundations = [{'seed':s,'status':'complete','updates':512,'checkpoint':'dummy.pt',
                        'checkpoint_sha256':digest,'adapter_fingerprint':'same'} for s in evaluation.SEEDS]
        runs=[]
        for s in evaluation.SEEDS:
            for a in evaluation.ARMS:
                steps=[0,256] if a=='frozen' else [0,64,128,192,256]
                runs.append({'seed':s,'arm':a,'status':'complete','primary_updates':0 if a=='frozen' else 256,
                    'extra_replay_updates':48 if a=='cms_extra_replay' else 0,
                    'sleep_adam_updates':96 if a=='cms_sleep' else 0,
                    'checkpoints':[{'base_step':n,'path':'dummy.pt','sha256':digest,
                                    'adapter_fingerprint':'same'} for n in steps]})
        return {'schema':'agnews-retention-training-seal-v2','complete':True,
                'records_sha256':evaluation.TRAIN_PACK_SHA,'official_test_opened':False,
                'protocol_sha256':evaluation.sha(root/'training-protocol.json'),
                'runs':runs,'foundation_checkpoints':foundations}

    def test_missing_or_incomplete_seal_blocks_before_test_read(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as d:
            root=Path(d)
            with self.assertRaises(FileNotFoundError): evaluation.require_seal(root)
            (root/'training-sealed.json').write_text('{"complete":false}')
            with self.assertRaises(RuntimeError): evaluation.require_seal(root)

    def test_complete_whole_block_and_tampered_budget(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as d:
            root=Path(d); seal=self.fixture(root)
            path=root/'training-sealed.json'
            path.write_text(json.dumps(seal))
            checked,_=evaluation.require_seal(root)
            self.assertEqual(len(checked['runs']),15)
            seal['runs'][-1]['extra_replay_updates']=47
            path.write_text(json.dumps(seal))
            with self.assertRaises(RuntimeError): evaluation.require_seal(root)

    def test_checkpoint_hash_and_common_foundation_guards(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as d:
            root=Path(d); seal=self.fixture(root)
            path=root/'training-sealed.json'
            seal['runs'][0]['checkpoints'][0]['adapter_fingerprint']='wrong'
            path.write_text(json.dumps(seal))
            with self.assertRaises(RuntimeError): evaluation.require_seal(root)
            seal=self.fixture(root);path.write_text(json.dumps(seal))
            (root/'dummy.pt').write_bytes(b'changed')
            with self.assertRaises(RuntimeError): evaluation.require_seal(root)


if __name__=='__main__':
    unittest.main()
