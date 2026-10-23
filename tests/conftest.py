"""Offline CPU acceptance guards, including charge-before-attempt accounting."""
import functools
import json
import os
from pathlib import Path
import socket
import time
import pytest
import torch
from nested_sleep.native import FunctionalCMS
from nested_sleep.optim import CMSController

ledger_path=Path(os.environ['RELEASE_BUDGET_FILE']) if os.environ.get('RELEASE_BUDGET_FILE') else None
started=time.monotonic()
ledger={'optimizer_attempts':0,'functional_cms_attempts':0,'controller_attempts':0,'limit':256}
if ledger_path and ledger_path.exists():
    ledger.update(json.loads(ledger_path.read_text(encoding='utf-8')))

def persist():
    ledger['last_session_seconds']=time.monotonic()-started
    if ledger_path:
        ledger_path.write_text(json.dumps(ledger,indent=2),encoding='utf-8')

def wrap(original,key):
    @functools.wraps(original)
    def call(*args,**kwargs):
        if time.monotonic()-started>3600:
            raise RuntimeError('CPU session wall-clock budget exceeded')
        if sum(ledger[k] for k in ('optimizer_attempts','functional_cms_attempts','controller_attempts'))>=256:
            raise RuntimeError('global toy update attempt cap reached')
        ledger[key]+=1;persist()
        return original(*args,**kwargs)
    return call

def denied(*args,**kwargs):
    raise RuntimeError('offline CPU tests prohibit network and CUDA initialization')

def pytest_sessionstart(session):
    torch.set_num_threads(1)
    torch.optim.AdamW.step=wrap(torch.optim.AdamW.step,'optimizer_attempts')
    torch.optim.SGD.step=wrap(torch.optim.SGD.step,'optimizer_attempts')
    FunctionalCMS.adapt=wrap(FunctionalCMS.adapt,'functional_cms_attempts')
    CMSController.collect=wrap(CMSController.collect,'controller_attempts')
    socket.socket.connect=denied
    socket.create_connection=denied
    torch.cuda._lazy_init=denied

def pytest_sessionfinish(session,exitstatus):
    assert not torch.cuda.is_initialized()
    persist()
