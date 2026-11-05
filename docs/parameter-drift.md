# Parameter-change diagnostic

`nested_sleep.parameter_drift.parameter_drift` compares two explicit mappings
of identically named floating-point parameter tensors. Capture a detached clone
*before* an update; `state_dict()` alone can share storage with the model.
For example, for a deliberately selected set of trainable parameters:

```python
from nested_sleep.parameter_drift import parameter_drift

names = [name for name, p in model.named_parameters() if p.requires_grad]
before = {name: p.detach().cpu().clone() for name, p in model.named_parameters()
          if name in names}
# Perform a separately authorized update here.
after = {name: p.detach().cpu().clone() for name, p in model.named_parameters()
         if name in names}
rows = parameter_drift(before, after)
```

Each row reports the baseline, current and difference L2 norms, relative change
`||after - before|| / ||before||`, element count and whether values differ.
The relative change is `None` for a zero-norm baseline. Names, shapes and dtypes
must match exactly; empty, nonfinite, non-floating and sparse tensors are
rejected. Norm calculations use CPU float64, one tensor at a time. The snapshots
can require significant memory; select a scope and device transfer deliberately.

The diagnostic performs no forward pass, training, checkpoint loading or file
writing. It does not infer why a parameter changed, whether an optimizer owned
it, whether learning succeeded, or whether a capability was retained. Record
snapshot provenance, parameter scope, and any gradient/optimizer evidence
separately. Equal before/after norms do not imply unchanged parameters. This
independent implementation was motivated by the weight-inspection discussion in
[obekt/HOPE-nested-learning](https://github.com/obekt/HOPE-nested-learning/commit/599244e);
no upstream source code is copied. The reference's weight observations are not
evidence about this project's models.
