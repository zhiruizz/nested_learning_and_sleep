import hashlib
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import banking_cil_common_v10 as b


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(value) for value in ids)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def classes77():
    return [f"intent_{index:02d}" for index in range(77)]


def row(text, intent_id, role="train", identity="r"):
    return {
        "id": sha(identity.encode()),
        "text": text,
        "intent_id": intent_id,
        "text_sha256": sha(text.encode("utf-8")),
        "normalized_text_sha256": sha(b.normalize_text(text).encode("utf-8")),
        "role": role,
    }


def orders():
    result = {}
    for seed in b.SEEDS:
        order = list(range(77))
        random.Random(f"banking77-cil-v10:{seed}").shuffle(order)
        result[str(seed)] = order
    return result


def manifest(files):
    classes = classes77()
    return {
        "schema": b.SCHEMA,
        "test_data_included": False,
        "derived_from_training_only": True,
        "num_classes": 77,
        "classes": classes,
        "source_label_to_id": {name: index for index, name in enumerate(classes)},
        "class_orders": orders(),
        "files": files,
        "source": {"dataset": "PolyAI/banking77 original", "revision": "9d081458ff52e53cf7e848f414e6e9344e4e6696",
                   "train_csv_sha256": "a" * 64, "train_csv_rows": 10003},
        "split_policy": {"grouped_before_split": True,
                         "normalized_text": "NFKC,casefold,whitespace-collapse; punctuation/underscore preserved",
                         "development_per_class": 10, "class_order_seeds": [17, 29, 43]},
    }


class BankingCILCommonV10Tests(unittest.TestCase):
    def test_segmented_natural_label_causal_mask_batch_and_gold_independent_prefix(self):
        labels = sorted(["a", "z" * 40] + [f"intent_{index:02d}" for index in range(75)])
        short_id, long_id = labels.index("a"), labels.index("z" * 40)
        text = "customer request " * 120
        first, second = row(text, short_id, identity="one"), row(text, long_id, identity="two")
        tokenizer = CharTokenizer()
        prompt = b.tokenize_prompt(text, tokenizer, labels, eos_token_id=2, max_length=110)
        encoded_a = b.tokenize_row(first, tokenizer, labels, eos_token_id=2, max_length=110)
        encoded_b = b.tokenize_row(second, tokenizer, labels, eos_token_id=2, max_length=110)
        self.assertEqual(encoded_a["input_ids"][:encoded_a["prompt_length"]], prompt["input_ids"])
        self.assertEqual(encoded_b["input_ids"][:encoded_b["prompt_length"]], prompt["input_ids"])
        self.assertEqual(encoded_a["input_ids"][:encoded_a["prompt_length"]], encoded_b["input_ids"][:encoded_b["prompt_length"]])
        self.assertEqual(encoded_a["input_ids"][0], ord("C"))  # no implicit BOS
        self.assertEqual(encoded_a["labels"][:encoded_a["prompt_length"]], [b.IGNORE_INDEX] * encoded_a["prompt_length"])
        self.assertEqual(encoded_a["labels"][-1], 2)
        self.assertEqual(tokenizer.decode(encoded_a["input_ids"][:-1]), b.build_prompt(text[:len(text)])[:0] +
                         tokenizer.decode(prompt["input_ids"]) + "a")
        self.assertEqual(b.decode_label_prefix(encoded_b["input_ids"][encoded_b["prompt_length"]:], tokenizer, 2), "z" * 40)
        batch = b.tokenize_batch([first, second], tokenizer, labels, eos_token_id=2, pad_token_id=0, max_length=110)
        self.assertEqual(batch["attention_mask"][0][-1], 0)
        self.assertEqual(batch["labels"][0][-1], b.IGNORE_INDEX)
        with self.assertRaises(ValueError):
            b.tokenize_row(first, tokenizer, labels, eos_token_id=2, max_length=20)

    def test_free_generation_normalization_invalids_and_77_class_task_matrix(self):
        labels = classes77()
        order = orders()["17"]
        result = b.score_predictions([labels[3].upper().replace("_", " "), "not-an-intent"], [3, 4], labels, order)
        self.assertEqual(result["correct"], 1)
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["invalid_predictions"], 1)
        self.assertFalse(result["candidate_constrained"])
        self.assertEqual(len(result["task_confusion"]), 7)
        self.assertTrue(all(len(row_values) == 7 for row_values in result["task_confusion"]))
        self.assertEqual(b.normalize_prediction("  INTENT  03\n"), "intent 03")
        self.assertEqual(len(b.task_class_ids(order, 6)), 11)
        self.assertEqual(set(class_id for task in range(7) for class_id in b.task_class_ids(order, task)), set(range(77)))

    def test_manifest_view_hash_checked_streaming_and_no_test_surface(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            root = Path(temp_dir)
            data = {
                "train": [row("Hello bank", 0, "train", "train")],
                "development": [row("Development case", 1, "development", "dev")],
            }
            files = {}
            for role, rows in data.items():
                path = root / f"{role}.jsonl"
                path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows), encoding="utf-8", newline="\n")
                files[role] = {"path": path.name, "sha256": b.sha256_file(path), "rows": len(rows)}
            payload = manifest(files)
            b.validate_manifest(payload)
            view = b.make_training_view(payload)
            self.assertFalse(hasattr(view, "test"))
            self.assertEqual(next(b.iter_role(view, "train", root))["intent_id"], 0)
            with self.assertRaises(ValueError):
                b.resolve_role_path(view, "test", root)
            changed = json.loads((root / "train.jsonl").read_text(encoding="utf-8"))
            changed["text"] = "edited"
            path = root / "train.jsonl"
            path.write_text(json.dumps(changed) + "\n", encoding="utf-8", newline="\n")
            new_meta = {"path": "train.jsonl", "sha256": b.sha256_file(path), "rows": 1}
            with self.assertRaisesRegex(ValueError, "text_sha256"):
                list(b.iter_jsonl(path, new_meta, view.classes, role="train"))

    def test_manifest_rejects_wrong_order_test_metadata_and_nontraining_file(self):
        base = manifest({"train": {"path": "train.jsonl", "sha256": "a" * 64, "rows": 1},
                         "development": {"path": "development.jsonl", "sha256": "b" * 64, "rows": 1}})
        bad = json.loads(json.dumps(base))
        bad["class_orders"]["29"][0], bad["class_orders"]["29"][1] = bad["class_orders"]["29"][1], bad["class_orders"]["29"][0]
        with self.assertRaisesRegex(ValueError, "locked Banking77 schedule"):
            b.validate_manifest(bad)
        bad = json.loads(json.dumps(base))
        bad["test"] = {"path": "test.jsonl"}
        with self.assertRaisesRegex(ValueError, "must not contain test"):
            b.validate_manifest(bad)
        bad = json.loads(json.dumps(base))
        bad["files"]["test"] = {"path": "test.jsonl", "sha256": "c" * 64, "rows": 1}
        with self.assertRaisesRegex(ValueError, "only train and development"):
            b.validate_manifest(bad)
        bad = json.loads(json.dumps(base))
        bad["files"]["train"]["path"] = "..\\test.jsonl"
        with self.assertRaisesRegex(ValueError, "stay within the pack root"):
            b.validate_manifest(bad)

    def test_task_schedule_and_partition_enforce_group_disjointness(self):
        order = orders()["17"]
        rows = [{"id": sha(f"{index}".encode()), "intent_id": index} for index in range(77)]
        for task_id in range(7):
            selected = list(b.iter_task_rows(rows, order, task_id))
            self.assertEqual(len(selected), 11)
            self.assertEqual({row["intent_id"] for row in selected}, set(b.task_class_ids(order, task_id)))
        self.assertEqual(b.shuffled_ids(rows, 17, 0), b.shuffled_ids(rows, 17, 0))
        train = [row("A phrase", 0, "train", "t")]
        dev = [row(" a   PHRASE ", 1, "development", "d")]
        with self.assertRaisesRegex(ValueError, "normalized text crosses"):
            b.validate_role_partition(train, dev)


if __name__ == "__main__":
    unittest.main()
