# Per-seed endpoint gates

Fixed gate definitions are applied to train/validation only; all three seeds are retained.

| Arm | Seed | Old gate | New gate | Joint gate |
|---|---:|---|---|---|
| current_receiver | 17 | FAIL | PASS | FAIL |
| current_receiver | 29 | FAIL | FAIL | FAIL |
| current_receiver | 43 | FAIL | PASS | FAIL |
| current_shared | 17 | FAIL | PASS | FAIL |
| current_shared | 29 | FAIL | FAIL | FAIL |
| current_shared | 43 | FAIL | PASS | FAIL |
| split_receiver | 17 | PASS | FAIL | FAIL |
| split_receiver | 29 | PASS | FAIL | FAIL |
| split_receiver | 43 | FAIL | PASS | FAIL |
| split_shared | 17 | PASS | FAIL | FAIL |
| split_shared | 29 | PASS | FAIL | FAIL |
| split_shared | 43 | PASS | PASS | PASS |
| split_narrow_receiver | 17 | FAIL | PASS | FAIL |
| split_narrow_receiver | 29 | FAIL | FAIL | FAIL |
| split_narrow_receiver | 43 | FAIL | PASS | FAIL |
| split_narrow_shared | 17 | FAIL | FAIL | FAIL |
| split_narrow_shared | 29 | FAIL | FAIL | FAIL |
| split_narrow_shared | 43 | FAIL | PASS | FAIL |
| gold_receiver | 17 | PASS | PASS | PASS |
| gold_receiver | 29 | PASS | FAIL | FAIL |
| gold_receiver | 43 | PASS | PASS | PASS |
| gold_shared | 17 | PASS | FAIL | FAIL |
| gold_shared | 29 | PASS | PASS | PASS |
| gold_shared | 43 | FAIL | PASS | FAIL |
| new_only_receiver | 17 | FAIL | PASS | FAIL |
| new_only_receiver | 29 | FAIL | PASS | FAIL |
| new_only_receiver | 43 | FAIL | PASS | FAIL |
| new_only_shared | 17 | FAIL | PASS | FAIL |
| new_only_shared | 29 | FAIL | PASS | FAIL |
| new_only_shared | 43 | FAIL | PASS | FAIL |
