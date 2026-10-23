"""Executable Sleep lifecycle, explicit assumptions in docs/paper_to_code.md.

Real autoregressive GKD/LTI/dream samples; immutable teacher, prospective student,
isolated LoRA trials, reward-filtered generator updates, checkpointable staging.
Local exact-output semantic judge is explicitly a proxy, not paper semantic RM.
"""
from copy import deepcopy
import hashlib
import random
import torch
from .lm import isolated_clone, immutable_versions
from torch.nn import functional as F
from .dreaming import select_dreams
from .objectives import edit_reward


def capture_rng():
    return {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'python': random.getstate()}


def restore_rng(state):
    torch.set_rng_state(state['cpu'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])
    random.setstate(state['python'])


def checked_state(model):
    # Immutable Llama backbone is identified by its pinned asset manifest, while
    # mutable adapters/readout are hashed here. Native model hashes all tensors.
    state = model.adapter_state_dict() if hasattr(model, 'adapter_state_dict') else model.state_dict()
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def fingerprint(model):
    h = hashlib.sha256()
    for name, value in sorted(checked_state(model).items()):
        h.update(name.encode())
        h.update(str((value.shape, value.dtype)).encode())
        h.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def continuation_loss(model, sequence, prefix_length):
    logits = model.logits(sequence)[:, prefix_length - 1:-1].float()
    targets = sequence[:, prefix_length:]
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


class ExactOutputSemanticProxy:
    """Frozen binary exact-equivalence proxy, deliberately labelled PARTIAL."""
    name = 'exact_output_proxy_NOT_paper_semantic_RM'
    def __call__(self, reference, candidate):
        return float(reference == candidate)


class SleepSession:
    def __init__(self, live, sender, prospective, *, seed=17, lr=.01, dream_tokens=4,
                 semantic_judge=None, share_backbone=False, reject_zero_gradient=False):
        if not 0 <= sender < len(live.level_descriptors()) - 1:
            raise ValueError('terminal sender policy is UNSPECIFIED; no slower receiver exists')
        self.live = live
        self.before = fingerprint(live)
        self.share_backbone = share_backbone
        self.reject_zero_gradient = reject_zero_gradient
        self.base_versions = immutable_versions(live) if share_backbone else {}
        self.teacher = isolated_clone(live, share_backbone=share_backbone).eval().requires_grad_(False)
        self.student = isolated_clone(live, share_backbone=share_backbone).eval().requires_grad_(False)
        self.sender, self.lr, self.dream_tokens = sender, lr, dream_tokens
        names = dict(self.student.named_parameters())
        expected_names = set(self.student.level_descriptors()[sender]['base'])
        if set(prospective) != expected_names:
            raise ValueError('prospective update must exactly match sender base scope')
        with torch.no_grad():
            for name, value in prospective.items():
                if not torch.isfinite(value).all():
                    raise ValueError('nonfinite prospective update')
                names[name].copy_(value)
        levels = self.student.level_descriptors()
        self.reset_slots = self.student.get_submodule(levels[sender]['bank']).reset_active()
        receiver = self.student.get_submodule(levels[sender + 1]['bank'])
        self.slot = receiver.activate()
        self.new_slot = receiver.slots[self.slot]
        self.new_slot.requires_grad_(True)
        self.allowed = {levels[sender + 1]['bank'] + f'.slots.{self.slot}.a',
                        levels[sender + 1]['bank'] + f'.slots.{self.slot}.b'}
        self.initial_student = checked_state(self.student)
        self.optimizer = torch.optim.AdamW([self.new_slot.a, self.new_slot.b], lr=lr, weight_decay=0.)
        device = next(live.parameters()).device
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.seed, self.stage, self.steps = seed, 'consolidate', 0
        self.records = []
        self.teacher_data = []
        self.step_observer = None
        self.semantic_judge = semantic_judge or ExactOutputSemanticProxy()
        self.work = {'generated_tokens': 0, 'distill_tokens': 0, 'lti_tokens': 0,
                     'trial_tokens': 0, 'policy_tokens': 0}

    def _optimize(self, loss, kind, tokens):
        if tokens <= 0 or not loss.requires_grad:
            raise RuntimeError('no effective token/gradient path in ' + kind)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite ' + kind + ' loss')
        if self.step_observer is not None:
            self.step_observer(self, 'before', loss, kind, tokens)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        before = [p.detach().clone() for p in (self.new_slot.a, self.new_slot.b)]
        gradient = [float(p.grad.detach().float().norm()) if p.grad is not None else None
                    for p in (self.new_slot.a, self.new_slot.b)]
        grad_norm = float(torch.nn.utils.clip_grad_norm_([self.new_slot.a, self.new_slot.b], 1.))
        if self.reject_zero_gradient and grad_norm == 0. and abs(float(loss.detach())) > 1e-6:
            self.optimizer.zero_grad(set_to_none=True)
            self.stage = 'failed'
            raise RuntimeError('nonzero objective but receiving expert gradient is zero; audit loss-position routing')
        self.optimizer.step()
        if self.step_observer is not None:
            self.step_observer(self, 'after', loss, kind, tokens)
        self.optimizer.zero_grad(set_to_none=True)
        delta = [float((p.detach().float() - b.float()).norm())
                 for p, b in zip((self.new_slot.a, self.new_slot.b), before)]
        self.records.append({'kind': kind, 'loss': float(loss.detach()), 'grad_norm': grad_norm, 'tokens': tokens,
                             'parameter_grad_norms': gradient, 'parameter_delta_norms': delta})

    def teacher_forced_step(self, sequence, prefix_length):
        """Independent fixed-sequence distillation control; same forward-KL objective."""
        if self.stage != 'consolidate' or not 1 <= prefix_length < sequence.shape[1]:
            raise ValueError('teacher-forced distillation needs nonempty continuation in consolidation')
        with torch.no_grad():
            p = self.teacher.logits(sequence)[:, prefix_length - 1:-1].float().log_softmax(-1)
        q = self.student.logits(sequence)[:, prefix_length - 1:-1].float().log_softmax(-1)
        loss = (p.exp() * (p - q)).sum(-1).mean()
        self.loss_context = {'sequence': sequence.detach(), 'prefix_length': prefix_length, 'teacher_logp': p.detach()}
        self._optimize(loss, 'teacher_forced_kl', sequence.shape[1] - prefix_length)
        self.work['distill_tokens'] += sequence.shape[1] - prefix_length

    def seed_step(self, prefix, *, owner=None):
        if self.stage != 'consolidate':
            raise RuntimeError('seeding after consolidation was closed')
        if owner not in (None, 'teacher', 'student_on_policy'):
            raise ValueError('unknown GKD sampling owner')
        if owner == 'teacher' or (owner is None and self.steps % 2 == 0):
            sequence, _ = self.teacher.sample(prefix, self.dream_tokens, generator=self.generator)
            self.teacher_data.append(sequence.detach().clone())
            owner = 'teacher'
        else:
            sequence, _ = self.student.sample(prefix, self.dream_tokens, generator=self.generator)
            owner = 'student_on_policy'
        n = prefix.shape[1]
        with torch.no_grad():
            p = self.teacher.logits(sequence)[:, n - 1:-1].float().log_softmax(-1)
        q = self.student.logits(sequence)[:, n - 1:-1].float().log_softmax(-1)
        loss = (p.exp() * (p - q)).sum(-1).mean()
        self.loss_context = {'sequence': sequence.detach(), 'prefix_length': n, 'teacher_logp': p.detach()}
        self._optimize(loss, 'gkd_' + owner, self.dream_tokens)
        self.work['generated_tokens'] += self.dream_tokens
        self.work['distill_tokens'] += self.dream_tokens
        self.steps += 1

    def lti_step(self, prefix):
        if self.stage != 'consolidate':
            raise RuntimeError('LTI outside consolidation')
        teacher_sequence, _ = self.teacher.sample(prefix, self.dream_tokens, generator=self.generator)
        cut = int(torch.randint(1, self.dream_tokens, (), device=prefix.device, generator=self.generator))
        prefix_length = prefix.shape[1] + cut
        start = teacher_sequence[:, :prefix_length]
        student_sequence, _ = self.student.sample(start, self.dream_tokens - cut, generator=self.generator)
        reference = teacher_sequence[0, prefix_length:].tolist()
        candidate = student_sequence[0, prefix_length:].tolist()
        semantic = self.semantic_judge(reference, candidate)
        reward = .5 * semantic + .5 * edit_reward(reference, candidate, raw_distance_threshold=len(reference))
        with torch.no_grad():
            p = self.teacher.logits(student_sequence)[:, prefix_length - 1:-1].float().log_softmax(-1)
        q = self.student.logits(student_sequence)[:, prefix_length - 1:-1].float().log_softmax(-1)
        kl = (p.exp() * (p - q)).sum(-1).mean()
        log_prob = q.gather(-1, student_sequence[:, prefix_length:, None]).squeeze(-1).mean()
        loss = .5 * kl - .5 * reward * log_prob
        self.loss_context = {'sequence': student_sequence.detach(), 'prefix_length': prefix_length,
                             'teacher_logp': p.detach(), 'reward': reward}
        self._optimize(loss, 'lti_reward_plus_kl', len(candidate))
        self.records[-1].update({'reward': reward, 'semantic': semantic, 'raw_reference': reference,
                                 'sampled_candidate': candidate, 'prefix_cut': cut})
        self.work['generated_tokens'] += self.dream_tokens + len(candidate)
        self.work['lti_tokens'] += len(candidate)

    def finish_consolidation(self):
        if self.stage != 'consolidate':
            raise RuntimeError('consolidation already closed')
        self.stage = 'dream'

    def dream(self, prefix, score_training_task, *, candidates=3, top_k=1, random_count=1,
              trial_steps=1, policy_steps=1):
        if self.stage != 'dream':
            raise RuntimeError('dreaming requires completed consolidation')
        base_hash = fingerprint(self.student)
        sequences, scores, routes = [], [], []
        for _ in range(candidates):
            seq, trace = self.student.sample(prefix, self.dream_tokens, generator=self.generator, dream=True)
            loss = continuation_loss(self.student, seq, prefix.shape[1])
            grads = torch.autograd.grad(loss, [self.new_slot.a, self.new_slot.b], allow_unused=True)
            scores.append(sum(float(g.detach().float().square().sum()) for g in grads if g is not None) ** .5)
            sequences.append(seq.detach())
            routes.append(trace)
        self.work['generated_tokens'] += candidates * self.dream_tokens
        selected = select_dreams(scores, top_k=top_k, random_count=random_count, seed=self.seed)
        baseline = float(score_training_task(self.student))
        trials, accepted = [], []
        for index in selected:
            trial = isolated_clone(self.student, share_backbone=self.share_backbone).eval().requires_grad_(False)
            policy = trial.policy_parameters()
            for p in policy:
                p.requires_grad_(True)
            opt = torch.optim.SGD(policy, lr=self.lr)
            start_hash = fingerprint(trial)
            for _ in range(trial_steps):
                opt.zero_grad(set_to_none=True)
                loss = continuation_loss(trial, sequences[index], prefix.shape[1])
                loss.backward()
                opt.step()
            after = float(score_training_task(trial))
            reward = int(after > baseline)
            trials.append({'index': index, 'reward': reward, 'baseline': baseline, 'after': after,
                           'initial_hash': start_hash, 'shared_initial_hash': base_hash})
            if reward:
                accepted.append(index)
            self.work['trial_tokens'] += trial_steps * self.dream_tokens
            del trial, opt
        assert fingerprint(self.student) == base_hash, 'isolated trials mutated shared model'
        policy = self.student.policy_parameters()
        self.student.requires_grad_(False)
        for p in policy:
            p.requires_grad_(True)
        opt = torch.optim.SGD(policy, lr=self.lr)
        for _ in range(policy_steps):
            for index in accepted:
                opt.zero_grad(set_to_none=True)
                loss = continuation_loss(self.student, sequences[index], prefix.shape[1])
                loss.backward()
                opt.step()
                self.work['policy_tokens'] += self.dream_tokens
        self.student.requires_grad_(False)
        self.allowed.update(self.student.policy_names())
        self.records.append({'kind': 'dreaming', 'generated': [s[0].tolist() for s in sequences],
                             'routes': routes, 'gradient_scores': scores, 'selected': selected,
                             'trials': trials, 'accepted': accepted,
                             'policy_updated': bool(accepted and policy_steps)})
        self.stage = 'ready'

    def commit(self):
        if self.stage not in ['ready', 'dream']:
            raise RuntimeError('cannot commit an unfinished or closed transaction')
        if fingerprint(self.live) != self.before or fingerprint(self.teacher) != self.before:
            raise RuntimeError('teacher/live changed before commit')
        if self.share_backbone and immutable_versions(self.live) != self.base_versions:
            raise RuntimeError('shared immutable backbone modified')
        now = checked_state(self.student)
        for name, value in now.items():
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise RuntimeError('nonfinite candidate')
            if name not in self.allowed and not torch.equal(value, self.initial_student[name]):
                raise RuntimeError('frozen state modified: ' + name)
        self.stage = 'committed'
        self.student.requires_grad_(False)
        return self.student

    def abort(self):
        if self.stage in ['committed', 'aborted']:
            raise RuntimeError('transaction already closed')
        self.stage = 'aborted'

    def state_dict(self):
        return {'schema': 1, 'sender': self.sender, 'slot': self.slot, 'before': self.before,
                'lr': self.lr, 'dream_tokens': self.dream_tokens, 'seed': self.seed,
                'teacher': self.teacher.state_dict(), 'student': self.student.state_dict(),
                'initial_student': self.initial_student, 'allowed': sorted(self.allowed),
                'optimizer': self.optimizer.state_dict(), 'generator': self.generator.get_state(),
                'stage': self.stage, 'steps': self.steps, 'records': self.records,
                'teacher_data': self.teacher_data, 'work': self.work, 'reset_slots': self.reset_slots,
                'rng': capture_rng(), 'share_backbone': self.share_backbone,
                'reject_zero_gradient': self.reject_zero_gradient}

    @classmethod
    def restore(cls, live, state, *, mutable_only=False):
        if state['schema'] != 1 or fingerprint(live) != state['before']:
            raise ValueError('checkpoint schema or live snapshot mismatch')
        # Construct without re-running prospective updates, expansion or reset.
        obj = cls.__new__(cls)
        obj.live = live
        obj.share_backbone = state.get('share_backbone', False)
        obj.reject_zero_gradient = state.get('reject_zero_gradient', False)
        obj.base_versions = immutable_versions(live) if obj.share_backbone else {}
        obj.teacher = isolated_clone(live, share_backbone=obj.share_backbone).eval().requires_grad_(False)
        obj.student = isolated_clone(live, share_backbone=obj.share_backbone).eval().requires_grad_(False)
        if mutable_only:
            expected=set(live.adapter_state_dict())
            if any(set(state[key])!=expected for key in ('teacher','student')):
                raise ValueError('compact checkpoint mutable scope mismatch')
        obj.teacher.load_state_dict(state['teacher'],strict=not mutable_only)
        obj.student.load_state_dict(state['student'],strict=not mutable_only)
        for key in ['sender', 'slot', 'before', 'lr', 'dream_tokens', 'seed', 'initial_student',
                    'stage', 'steps', 'records', 'teacher_data', 'work', 'reset_slots']:
            setattr(obj, key, deepcopy(state[key]))
        obj.allowed = set(state['allowed'])
        receiver = obj.student.get_submodule(obj.student.level_descriptors()[obj.sender + 1]['bank'])
        obj.new_slot = receiver.slots[obj.slot]
        obj.new_slot.requires_grad_(True)
        obj.optimizer = torch.optim.AdamW([obj.new_slot.a, obj.new_slot.b], lr=obj.lr, weight_decay=0.)
        obj.optimizer.load_state_dict(state['optimizer'])
        obj.generator = torch.Generator(device=next(live.parameters()).device)
        obj.generator.set_state(state['generator'].cpu())
        obj.semantic_judge = ExactOutputSemanticProxy()
        obj.step_observer = None
        restore_rng(state['rng'])
        return obj
