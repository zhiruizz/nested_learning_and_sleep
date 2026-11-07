"""Archive only the source/config/review metadata of the bounded developer probe."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
preflight = root / "outputs/clinc-v8-preparation/controlled-probe-local-preflight.json"
record = json.loads(preflight.read_text(encoding="utf-8"))
out = root / "outputs/clinc-v8-controlled-probe-frozen-source-r2"
out.mkdir(exist_ok=False)
archived = []
for original, expected in record["source_hashes"].items():
    path = Path(original)
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise ValueError(f"source changed after preflight: {path.name}")
    target = out / path.name
    if target.exists():
        raise FileExistsError(target)
    target.write_bytes(raw)
    archived.append({"original_path":original,"archive_file":target.name,"sha256":actual})
config = root / "configs/clinc_cil_v8_controlled_probe_seed17.json"
if hashlib.sha256(config.read_bytes()).hexdigest() != record["config_file_sha256"]:
    raise ValueError("config changed after preflight")
shutil.copyfile(config, out / config.name)
shutil.copyfile(preflight, out / "local-preflight.json")
review = {"schema":"clinc-v8-independent-review-record-v1",
          "recorded_utc":datetime.now(timezone.utc).isoformat(),
          "reviewer":"/root/v4_boundary_helper", "outcome":"no_start_blocker",
          "reviewer_conclusion":"可启动当前冻结的开发 probe；无阻断。",
          "driver_sha256":"693f0edf7352c3e874613dd128ca4398035d5064d4a606c440e526ddb386151d",
          "review_addendum":"The initial launch failed before model load due to local torch shadowing. The independent reviewer verified the loaded_torch identity fix and the unchanged config/test hashes; a separate actual local startup sentinel qualification passed with no CUDA/model initialization.",
          "review_scope":"source/config/test hashes, qualified checkpoint/data binding, identical forks, schedule, nested lifecycle, paired replay, gradient scope, route masks, CP isolation, failure/accounting and no test input",
          "nonblocking_caveat":"gold CE increments the internal gkd_calls lifecycle bucket; event kind/uses_gold/physical Adam distinguish the objective",
          "preflight_sha256":hashlib.sha256(preflight.read_bytes()).hexdigest(),
          "source_lock_sha256":record["source_lock_sha256"],
          "config_file_sha256":record["config_file_sha256"], "sources":archived}
(out / "source-review-lock.json").write_text(json.dumps(review, indent=2, ensure_ascii=False)+"\n",encoding="utf-8")
print(json.dumps({"archived_source_files":len(archived),"source_lock_sha256":record["source_lock_sha256"]}))
