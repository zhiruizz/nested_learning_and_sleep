# AGNews v3 aggregate forensics

Scope: public aggregate JSON and report only. No raw examples, prompt/label data, model files, GPU, or training were accessed. Denominators are old/new/all = 512/512/1024; each class = 256.

## Endpoint comparison

Mean full-vocabulary two-token EM across the three seeds; percentages. Descriptive only.

| Arm | Old | New | All | Mean answer NLL (all) |
|---|---:|---:|---:|---:|
| cms | 0.00% | 84.70% | 42.35% | 2.3956 |
| cms_sleep | 0.00% | 77.86% | 38.93% | 2.9856 |
| cms_extra_replay | 49.48% | 69.79% | 59.64% | 0.6408 |
| sft | 0.00% | 88.61% | 44.30% | 2.4342 |
| frozen | 93.68% | 0.00% | 46.84% | 3.2621 |

CMS+Sleep minus CMS: old +0.00 pp; new -6.84 pp; all -3.42 pp.

## Sleep event diagnostics

| Seed | Events | GKD marker hits | LTI teacher marker hits | Natural GKD positions | Natural LTI positions | Positive-gradient events | Commits | Final active slots |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 17 | 16 | 248/256 | 128/128 | 357/512 | 91/128 | 16/16 | 16/16 | [0, 16] |
| 29 | 16 | 255/256 | 128/128 | 318/512 | 72/128 | 16/16 | 16/16 | [0, 16] |
| 43 | 16 | 244/256 | 128/128 | 358/512 | 87/128 | 16/16 | 16/16 | [0, 16] |

Executed mixed-receiver coverage was 1.0 for all 48 events. Natural receiver hit rates varied (per-seed event means 70.0%, 60.9%, 69.5%; overall range 0 to 100%). Each seed postcommit active-slot sequence progressed [0,1], [0,2], ... [0,16].

## Checkpoint trajectory

Mean full-vocabulary EM by checkpoint; each arm is averaged over three seeds.

| Arm / step | Old | New | All | Marker accuracy (all) |
|---|---:|---:|---:|---:|
| cms / 0 | 93.68% | 0.00% | 46.84% | 100.00% |
| cms / 256 | 0.00% | 71.22% | 35.61% | 100.00% |
| cms / 512 | 0.00% | 74.48% | 37.24% | 100.00% |
| cms / 768 | 0.00% | 80.21% | 40.10% | 100.00% |
| cms / 1024 | 0.00% | 84.70% | 42.35% | 100.00% |
| cms_sleep / 0 | 93.68% | 0.00% | 46.84% | 100.00% |
| cms_sleep / 256 | 0.00% | 61.72% | 30.86% | 99.45% |
| cms_sleep / 512 | 0.00% | 61.20% | 30.60% | 100.00% |
| cms_sleep / 768 | 0.00% | 86.13% | 43.07% | 100.00% |
| cms_sleep / 1024 | 0.00% | 77.86% | 38.93% | 100.00% |
| cms_extra_replay / 0 | 93.68% | 0.00% | 46.84% | 100.00% |
| cms_extra_replay / 256 | 68.75% | 37.63% | 53.19% | 100.00% |
| cms_extra_replay / 512 | 54.75% | 63.67% | 59.21% | 100.00% |
| cms_extra_replay / 768 | 54.82% | 73.11% | 63.96% | 100.00% |
| cms_extra_replay / 1024 | 49.48% | 69.79% | 59.64% | 100.00% |

## Evidence and limits

- Marker accuracy remains 1.0 at the Sleep endpoint, while class prediction fails for old tasks: class0/class1 EM is 0 in every seed; new class2 averages 95.31%, while class3 averages 60.42% and varies more by seed. This marker-vs-class split locates failure downstream of marker recognition; it does not identify why it happened.
- CMS and Sleep both reach 0 old-task EM by step 256 from a shared 93.68% starting point. Sleep new-task EM varies across checkpoints (61.7%, 61.2%, 86.1%, 77.9% at steps 256 to 1024), so the endpoint follows a non-monotonic trajectory. CMS rises from 71.2% to 84.7% at those checkpoints.
- Natural receiver coverage is not complete: per-seed aggregate hits are listed above and per-event rates range from 0 to 1. All 48 events have positive-gradient flags and commits, but GKD sampled-marker hits (teacher/student rollouts) are short of total in some events. Executed mix covered every loss position, so incomplete natural coverage does not establish blocked receiving gradients in v3 or explain its endpoint gap.
- CMS, CMS+Sleep, and SFT each have 0% old-task endpoint EM across all three seeds. CMS+Sleep has 77.86% new EM versus CMS 84.70%; this small-seed aggregate is descriptive and does not support causal or significance claims.
- CMS+extra-replay retains old accuracy better on average but is heterogeneous across seeds; inspect the per-seed and checkpoint rows in the JSON. It is a comparison arm, not a causal control for Sleep.
- The public report scopes the holdout as fresh within-AGNews-train rather than external/pretraining-independent confirmation and explicitly says this is not a full Sleep/HOPE reproduction.

## Files

`aggregate-forensics.json` contains all 66 checkpoint rows, the 48 per-event rows, per-seed endpoint metrics, arm means and checkpoint trajectories.
