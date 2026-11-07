"""Pure-Python dataset, tokenization, and metric helpers for Banking77 CIL.

The module supports 77 canonical labels arranged as seven tasks of eleven
classes. It does not import model or tokenizer frameworks and has no test-set
reader. Callers provide a tokenizer implementing ``encode`` and ``decode``.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
import unicodedata
from pathlib import Path, PureWindowsPath
from typing import Iterable, Iterator, Mapping, Sequence

SCHEMA = "banking-cil-train-dev-manifest-v10"
SEEDS = (17, 29, 43)
TASKS = 7
CLASSES_PER_TASK = 11
NUM_CLASSES = 77
PROMPT_TEMPLATE = "Classify the intent of this user request.\nRequest: {text}\nIntent:"
PROMPT_HEAD = "Classify the intent of this user request.\nRequest: "
PROMPT_TAIL = "\nIntent:"
IGNORE_INDEX = -100


@dataclass(frozen=True)
class TrainingView:
    """Validated train/development view with no test field or test path."""

    schema: str
    classes: tuple[str, ...]
    class_orders: Mapping[str, tuple[int, ...]]
    train: Mapping[str, object]
    development: Mapping[str, object]
    source_sha256: str


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_text(text: str) -> str:
    """The pack's duplicate-group normalization; punctuation and underscores stay."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _is_hex64(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def _validate_classes(classes: Sequence[str]) -> tuple[str, ...]:
    values = tuple(classes)
    if len(values) != NUM_CLASSES or len(set(values)) != NUM_CLASSES:
        raise ValueError("expected 77 unique canonical intent names")
    if any(not isinstance(value, str) or not value or value != value.strip() for value in values):
        raise ValueError("class names must be nonempty canonical strings")
    if any(value != unicodedata.normalize("NFKC", value).casefold() for value in values):
        raise ValueError("class names must be NFKC casefold canonical strings")
    if values != tuple(sorted(values)):
        raise ValueError("canonical classes must be lexicographically sorted")
    normalized = [normalize_prediction(value) for value in values]
    if len(set(normalized)) != NUM_CLASSES:
        raise ValueError("canonical labels collide after underscore/space normalization")
    return values


def _expected_order(seed: int) -> tuple[int, ...]:
    order = list(range(NUM_CLASSES))
    random.Random(f"banking77-cil-v10:{seed}").shuffle(order)
    return tuple(order)


def validate_manifest(manifest: Mapping) -> None:
    if manifest.get("schema") != SCHEMA:
        raise ValueError("manifest schema mismatch")
    if manifest.get("num_classes") != NUM_CLASSES:
        raise ValueError("manifest must declare 77 classes")
    classes = _validate_classes(manifest.get("classes", ()))
    if manifest.get("test_data_included") is not False:
        raise ValueError("training manifest must explicitly exclude test data")
    if "test" in manifest:
        raise ValueError("training manifest must not contain test metadata")
    if manifest.get("derived_from_training_only") is not True:
        raise ValueError("training manifest must declare its training-only source")

    orders = manifest.get("class_orders")
    if not isinstance(orders, dict) or set(orders) != {str(seed) for seed in SEEDS}:
        raise ValueError("class_orders must define exactly seeds 17, 29, and 43")
    for seed in SEEDS:
        order = orders[str(seed)]
        if not isinstance(order, list) or len(order) != NUM_CLASSES or any(type(value) is not int for value in order):
            raise ValueError(f"seed {seed}: class order must contain 77 integer IDs")
        if tuple(order) != _expected_order(seed):
            raise ValueError(f"seed {seed}: class order differs from the locked Banking77 schedule")

    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {"train", "development"}:
        raise ValueError("manifest must contain only train and development file metadata")
    for role, meta in files.items():
        if not isinstance(meta, dict):
            raise ValueError(f"malformed {role} metadata")
        if not isinstance(meta.get("path"), str) or not meta["path"]:
            raise ValueError(f"{role} path missing")
        local_path = Path(meta["path"])
        windows_path = PureWindowsPath(meta["path"])
        if local_path.is_absolute() or windows_path.is_absolute() or windows_path.drive or ".." in local_path.parts or ".." in windows_path.parts:
            raise ValueError(f"{role} path must stay within the pack root")
        if not _is_hex64(meta.get("sha256")):
            raise ValueError(f"{role} SHA256 must be hex64")
        if type(meta.get("rows")) is not int or meta["rows"] < 0:
            raise ValueError(f"{role} row count must be a nonnegative integer")

    source = manifest.get("source")
    if not isinstance(source, dict) or not _is_hex64(source.get("train_csv_sha256")):
        raise ValueError("source training CSV SHA256 is required")
    if source.get("train_csv_rows") != 10003 or not isinstance(source.get("revision"), str) or not source["revision"]:
        raise ValueError("source revision and expected training-row count are required")

    policy = manifest.get("split_policy")
    if not isinstance(policy, dict):
        raise ValueError("split_policy metadata is required")
    if policy.get("grouped_before_split") is not True:
        raise ValueError("normalized-text groups must be split before role assignment")
    if policy.get("normalized_text") != "NFKC,casefold,whitespace-collapse; punctuation/underscore preserved":
        raise ValueError("normalized-text policy mismatch")
    if policy.get("development_per_class") != 10:
        raise ValueError("development split must retain the locked ten examples per class")
    if policy.get("class_order_seeds") != list(SEEDS):
        raise ValueError("split policy class-order seeds mismatch")

    label_map = manifest.get("source_label_to_id")
    if not isinstance(label_map, dict) or not label_map:
        raise ValueError("source_label_to_id mapping is required")
    if any(type(value) is not int or not 0 <= value < NUM_CLASSES for value in label_map.values()):
        raise ValueError("source_label_to_id contains an out-of-range class ID")
    if set(label_map.values()) != set(range(NUM_CLASSES)):
        raise ValueError("source_label_to_id must cover all 77 canonical classes")
    canonical_by_id = {index: name for index, name in enumerate(classes)}
    if any(canonical_label(label) != canonical_by_id[index] for label, index in label_map.items()):
        raise ValueError("source label mapping does not match canonical class order")


def make_training_view(manifest: Mapping) -> TrainingView:
    """Validate a manifest and expose train/dev metadata only."""
    validate_manifest(manifest)
    return TrainingView(
        schema=SCHEMA,
        classes=tuple(manifest["classes"]),
        class_orders={str(seed): tuple(manifest["class_orders"][str(seed)]) for seed in SEEDS},
        train=dict(manifest["files"]["train"]),
        development=dict(manifest["files"]["development"]),
        source_sha256=canonical_json_sha256(manifest["source"]),
    )


def resolve_role_path(view: TrainingView, role: str, root: str | Path) -> Path:
    if role not in ("train", "development"):
        raise ValueError("training view only exposes train and development")
    return Path(root) / str(getattr(view, role)["path"])


def _validate_row(row: Mapping, classes: Sequence[str], expected_role: str | None = None) -> None:
    row_id = row.get("id")
    if not _is_hex64(row_id):
        raise ValueError("row id must be source-bound hex64")
    text = row.get("text")
    if not isinstance(text, str):
        raise ValueError("row text must be a string")
    intent_id = row.get("intent_id")
    if type(intent_id) is not int or not 0 <= intent_id < len(classes):
        raise ValueError("intent_id outside canonical class range")
    text_sha = row.get("text_sha256")
    if not _is_hex64(text_sha) or text_sha.lower() != hashlib.sha256(text.encode("utf-8")).hexdigest():
        raise ValueError("text_sha256 does not match original UTF-8 text")
    normalized_sha = row.get("normalized_text_sha256")
    expected_normalized = hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
    if not _is_hex64(normalized_sha) or normalized_sha.lower() != expected_normalized:
        raise ValueError("normalized_text_sha256 does not match locked text normalization")
    if expected_role is not None and row.get("role") != expected_role:
        raise ValueError("row role mismatch")
    if not isinstance(row.get("role"), str) or row["role"] not in ("train", "development"):
        raise ValueError("row role must be train or development")


def row_intent_id(row: Mapping, classes: Sequence[str] | None = None) -> int:
    value = row.get("intent_id")
    if type(value) is not int or not 0 <= value < NUM_CLASSES:
        raise ValueError("row lacks an in-range integer intent_id")
    if classes is not None and len(classes) != NUM_CLASSES:
        raise ValueError("classes must contain all 77 canonical labels")
    return value


def iter_jsonl(path: str | Path, metadata: Mapping, classes: Sequence[str], *, role: str) -> Iterator[dict]:
    """Verify exact file bytes before streaming and validating rows."""
    if role not in ("train", "development"):
        raise ValueError("only train and development JSONL are exposed")
    path = Path(path)
    if sha256_file(path) != metadata["sha256"]:
        raise ValueError(f"{role} file SHA256 mismatch")
    count = 0
    ids: set[str] = set()
    normalized_texts: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL line at {line_no}")
            row = json.loads(line)
            _validate_row(row, classes, role)
            row_id = row["id"]
            if row_id in ids:
                raise ValueError(f"duplicate {role} id: {row_id}")
            ids.add(row_id)
            normalized_text = row["normalized_text_sha256"]
            if normalized_text in normalized_texts:
                raise ValueError(f"duplicate normalized text in {role}: {normalized_text}")
            normalized_texts.add(normalized_text)
            count += 1
            yield row
    if count != metadata["rows"]:
        raise ValueError(f"{role} row count mismatch: {count} != {metadata['rows']}")


def iter_role(view: TrainingView, role: str, root: str | Path) -> Iterator[dict]:
    if role not in ("train", "development"):
        raise ValueError("training view only exposes train and development")
    metadata = getattr(view, role)
    return iter_jsonl(resolve_role_path(view, role, root), metadata, view.classes, role=role)


def validate_role_partition(train: Iterable[Mapping], development: Iterable[Mapping]) -> dict:
    """Check ID and normalized-text disjointness and return observed class counts."""
    ids: set[str] = set()
    text_hashes: set[str] = set()
    counts: dict[str, list[int]] = {"train": [0] * NUM_CLASSES, "development": [0] * NUM_CLASSES}
    totals: dict[str, int] = {}
    for role, rows in (("train", train), ("development", development)):
        total = 0
        local_ids: set[str] = set()
        for row in rows:
            row_id = row.get("id")
            text_hash = row.get("normalized_text_sha256")
            intent_id = row_intent_id(row)
            if not _is_hex64(row_id) or not _is_hex64(text_hash):
                raise ValueError("partition rows require valid row and normalized-text hashes")
            if row_id in ids or row_id in local_ids:
                raise ValueError("split row IDs overlap or repeat")
            if text_hash in text_hashes:
                raise ValueError("normalized text crosses train/development roles")
            local_ids.add(row_id)
            text_hashes.add(text_hash)
            counts[role][intent_id] += 1
            total += 1
        ids.update(local_ids)
        totals[role] = total
    return {"rows_by_role": totals, "class_counts": counts}


def task_for_class(class_id: int, class_order: Sequence[int]) -> int:
    if len(class_order) != NUM_CLASSES or type(class_id) is not int or class_id not in class_order:
        raise ValueError("class/order mismatch")
    return tuple(class_order).index(class_id) // CLASSES_PER_TASK


def task_class_ids(class_order: Sequence[int], task_id: int) -> tuple[int, ...]:
    if len(class_order) != NUM_CLASSES or type(task_id) is not int or not 0 <= task_id < TASKS:
        raise ValueError("invalid class order or task id")
    if set(class_order) != set(range(NUM_CLASSES)):
        raise ValueError("class order must be a permutation of 0..76")
    start = task_id * CLASSES_PER_TASK
    return tuple(class_order[start:start + CLASSES_PER_TASK])


def iter_task_rows(rows: Iterable[Mapping], class_order: Sequence[int], task_id: int,
                   classes: Sequence[str] | None = None) -> Iterator[Mapping]:
    allowed = set(task_class_ids(class_order, task_id))
    for row in rows:
        if row_intent_id(row, classes) in allowed:
            yield row


def shuffled_ids(rows: Iterable[Mapping], seed: int, task_id: int) -> list[str]:
    ids = [str(row["id"]) for row in rows]
    random.Random(f"banking-cil-v10:{seed}:{task_id}").shuffle(ids)
    return ids


def build_prompt(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be str")
    return PROMPT_TEMPLATE.format(text=text)


def _encode(tokenizer, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        output = tokenizer.encode(text, add_special_tokens=False)
    else:
        output = tokenizer(text, add_special_tokens=False)["input_ids"]
    if output and isinstance(output[0], list):
        output = output[0]
    return [int(value) for value in output]


def tokenize_prompt(text: str, tokenizer, classes: Sequence[str], eos_token_id: int,
                    max_length: int | None = None) -> dict:
    """Tokenize a gold-independent prompt with the historical segmented recipe.

    Prompt head, request text, and prompt tail are encoded separately without
    automatic special tokens, then concatenated directly. No BOS is inserted.
    Truncation applies only to request text and reserves the longest label+EOS.
    """
    classes = _validate_classes(classes)
    if not isinstance(text, str):
        raise TypeError("text must be str")
    head_ids = _encode(tokenizer, PROMPT_HEAD)
    body_ids = _encode(tokenizer, text)
    tail_ids = _encode(tokenizer, PROMPT_TAIL)
    label_lengths = [len(_encode(tokenizer, label)) for label in classes]
    max_label_tokens = max(label_lengths)
    if max_label_tokens < 1:
        raise ValueError("canonical label tokenized to an empty sequence")
    if max_length is not None:
        body_budget = int(max_length) - len(head_ids) - len(tail_ids) - max_label_tokens - 1
        if body_budget < 0:
            raise ValueError("max_length cannot fit prompt frame and longest label+EOS")
        body_ids = body_ids[:body_budget]
    prompt_ids = head_ids + body_ids + tail_ids
    return {"input_ids": prompt_ids, "attention_mask": [1] * len(prompt_ids), "prompt_length": len(prompt_ids)}


def tokenize_row(row: Mapping, tokenizer, classes: Sequence[str], eos_token_id: int,
                 max_length: int | None = None) -> dict:
    classes = _validate_classes(classes)
    _validate_row(row, classes)
    prefix = tokenize_prompt(row["text"], tokenizer, classes, eos_token_id, max_length)
    target_ids = _encode(tokenizer, classes[row_intent_id(row, classes)])
    if not target_ids:
        raise ValueError("canonical label tokenized to empty sequence")
    prompt_ids = prefix["input_ids"]
    input_ids = prompt_ids + target_ids + [int(eos_token_id)]
    labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids + [int(eos_token_id)]
    return {"id": row["id"], "intent_id": row["intent_id"], "prompt_length": len(prompt_ids),
            "input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def tokenize_batch(rows: Sequence[Mapping], tokenizer, classes: Sequence[str], eos_token_id: int,
                   pad_token_id: int, max_length: int | None = None) -> dict:
    if not rows:
        raise ValueError("empty batch")
    encoded = [tokenize_row(row, tokenizer, classes, eos_token_id, max_length) for row in rows]
    width = max(len(item["input_ids"]) for item in encoded)
    return {
        "ids": [item["id"] for item in encoded],
        "intent_ids": [item["intent_id"] for item in encoded],
        "input_ids": [item["input_ids"] + [int(pad_token_id)] * (width - len(item["input_ids"])) for item in encoded],
        "attention_mask": [item["attention_mask"] + [0] * (width - len(item["attention_mask"])) for item in encoded],
        "labels": [item["labels"] + [IGNORE_INDEX] * (width - len(item["labels"])) for item in encoded],
        "prompt_lengths": [item["prompt_length"] for item in encoded],
    }


def normalize_prediction(text: str, *, underscore_space_equivalent: bool = True) -> str:
    value = unicodedata.normalize("NFKC", str(text)).casefold().strip()
    if underscore_space_equivalent:
        value = value.replace("_", " ")
    return " ".join(value.split())


def canonical_label(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold().strip()


def decode_label_prefix(token_ids: Sequence[int], tokenizer, eos_token_id: int) -> str:
    values = list(token_ids)
    try:
        values = values[:values.index(int(eos_token_id))]
    except ValueError:
        pass
    return tokenizer.decode(values, skip_special_tokens=True)


def score_predictions(predictions: Sequence[str], gold_intent_ids: Sequence[int], classes: Sequence[str],
                      class_order: Sequence[int] | None = None) -> dict:
    if len(predictions) != len(gold_intent_ids):
        raise ValueError("prediction/gold length mismatch")
    classes = _validate_classes(classes)
    label_to_id = {normalize_prediction(value): index for index, value in enumerate(classes)}
    if class_order is not None and (len(class_order) != NUM_CLASSES or set(class_order) != set(range(NUM_CLASSES))):
        raise ValueError("class_order must be a permutation of 0..76")
    task_map = {class_id: task_for_class(class_id, class_order) for class_id in range(NUM_CLASSES)} if class_order is not None else None
    correct = invalid = 0
    per_class = {index: {"correct": 0, "total": 0} for index in range(NUM_CLASSES)}
    task_matrix = [[0 for _ in range(TASKS)] for _ in range(TASKS)]
    for prediction, gold in zip(predictions, gold_intent_ids):
        if type(gold) is not int or not 0 <= gold < NUM_CLASSES:
            raise ValueError("gold class ID outside range")
        parsed = label_to_id.get(normalize_prediction(prediction))
        is_correct = parsed == gold
        correct += int(is_correct)
        invalid += int(parsed is None)
        per_class[gold]["total"] += 1
        per_class[gold]["correct"] += int(is_correct)
        if task_map is not None and parsed is not None:
            task_matrix[task_map[gold]][task_map[parsed]] += 1
    total = len(predictions)
    return {
        "metric": "free_generation_micro_exact_normalized_label_accuracy",
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "invalid_predictions": invalid,
        "per_class": {str(index): {"correct": row["correct"], "total": row["total"],
                                   "accuracy": row["correct"] / row["total"] if row["total"] else None}
                      for index, row in per_class.items()},
        "task_confusion": task_matrix if task_map is not None else None,
        "candidate_constrained": False,
    }
