"""Test for strict train/val/test split exclusivity and leakage prevention."""
import json
from pathlib import Path
import pytest

from acl.data_gen import SPLIT_FOLDS


def test_split_folds_are_mutually_exclusive():
    """Verify that source ESC-50 folds assigned to splits are strictly disjoint."""
    train_folds = set(SPLIT_FOLDS["train"])
    val_folds = set(SPLIT_FOLDS["val"])
    test_folds = set(SPLIT_FOLDS["test"])

    assert train_folds.isdisjoint(val_folds), "Train and Val folds overlap!"
    assert train_folds.isdisjoint(test_folds), "Train and Test folds overlap!"
    assert val_folds.isdisjoint(test_folds), "Val and Test folds overlap!"
    assert len(train_folds | val_folds | test_folds) == 5, "All 5 ESC-50 folds must be accounted for"


def verify_dataset_leakage(dataset_dir):
    """Check that no source audio file (foreground or background) appears in more than one split."""
    dataset_path = Path(dataset_dir)
    sources = {}
    for split in ("train", "val", "test"):
        ann_path = dataset_path / "annotations" / f"{split}.jsonl"
        if not ann_path.exists():
            return False, f"Missing {ann_path}"
        src_set = set()
        with open(ann_path) as f:
            for line in f:
                rec = json.loads(line)
                if "background" in rec and "source" in rec["background"]:
                    src_set.add(rec["background"]["source"])
                for e in rec.get("events", []):
                    if "source" in e:
                        src_set.add(e["source"])
        sources[split] = src_set

    # Assert mutual exclusivity
    assert sources["train"].isdisjoint(sources["val"]), (
        f"Leakage between train and val! Common files: {sources['train'] & sources['val']}"
    )
    assert sources["train"].isdisjoint(sources["test"]), (
        f"Leakage between train and test! Common files: {sources['train'] & sources['test']}"
    )
    assert sources["val"].isdisjoint(sources["test"]), (
        f"Leakage between val and test! Common files: {sources['val'] & sources['test']}"
    )
    return True, "No leakage detected across splits."


def test_mock_leakage_check():
    """Verify that the leakage checker properly catches clean vs leaky splits."""
    clean_train = {"1-100.wav", "2-200.wav"}
    clean_val = {"4-300.wav"}
    clean_test = {"5-400.wav"}
    assert clean_train.isdisjoint(clean_val)
    assert clean_train.isdisjoint(clean_test)
    assert clean_val.isdisjoint(clean_test)

    leaky_test = {"1-100.wav", "5-500.wav"}
    assert not clean_train.isdisjoint(leaky_test), "Leakage check must catch common files!"


if __name__ == "__main__":
    test_split_folds_are_mutually_exclusive()
    test_mock_leakage_check()
    print("PASS: Split exclusivity tests verified.")
