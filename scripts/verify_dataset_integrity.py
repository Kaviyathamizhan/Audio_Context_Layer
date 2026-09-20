import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acl.classes import CLASSES, N_CLASSES
from acl.qa_engine import QAEngine


def verify_dataset(data_dir):
    data_path = Path(data_dir)
    print(f"=== Verifying Dataset Integrity: {data_path} ===")

    # 1. Leakage Check
    sources = {}
    all_scenes = set()
    scene_events = {}

    for split in ("train", "val", "test"):
        ann_path = data_path / "annotations" / f"{split}.jsonl"
        assert ann_path.exists(), f"Missing annotations for {split}"
        split_sources = set()
        split_scenes = set()

        with open(ann_path) as f:
            for line_idx, line in enumerate(f):
                rec = json.loads(line)
                sid = rec["scene_id"]
                assert sid not in all_scenes, f"Duplicate scene_id across splits: {sid}"
                assert sid not in split_scenes, f"Duplicate scene_id within {split}: {sid}"
                split_scenes.add(sid)
                all_scenes.add(sid)

                # Check background source
                assert "background" in rec and "source" in rec["background"], f"Missing background source in {sid}"
                split_sources.add(rec["background"]["source"])

                # Check events
                events = rec.get("events", [])
                assert len(events) >= 2, f"Scene {sid} has fewer than 2 events: {len(events)}"
                scene_events[sid] = events

                for e_idx, e in enumerate(events):
                    assert "source" in e, f"Missing event source in {sid} event {e_idx}"
                    split_sources.add(e["source"])

                    # Check onset < offset and [0, 10]
                    on, off = e["onset"], e["offset"]
                    assert 0.0 <= on < off <= 10.0, f"Invalid event boundary in {sid}: onset={on}, offset={off}"
                    assert e["class"] in CLASSES, f"Invalid class {e['class']} in {sid}"

        sources[split] = split_sources
        print(f"  [{split}] Verified {len(split_scenes)} scenes, {len(split_sources)} unique source clips.")

    # Strict disjointness
    assert sources["train"].isdisjoint(sources["val"]), f"Train/Val leakage! Common: {sources['train'] & sources['val']}"
    assert sources["train"].isdisjoint(sources["test"]), f"Train/Test leakage! Common: {sources['train'] & sources['test']}"
    assert sources["val"].isdisjoint(sources["test"]), f"Val/Test leakage! Common: {sources['val'] & sources['test']}"
    print("  [LEAKAGE CHECK] PASSED: train, val, and test source clips are strictly disjoint.")

    # 2. QA Scene ID resolution & Ground-Truth Consistency
    engine = QAEngine()
    total_qa = 0
    mismatches = []

    for split in ("train", "val", "test"):
        qa_path = data_path / "qa" / f"{split}.jsonl"
        assert qa_path.exists(), f"Missing QA for {split}"

        with open(qa_path) as f:
            for q_idx, line in enumerate(f):
                q = json.loads(line)
                sid = q["scene_id"]
                assert sid in all_scenes, f"QA scene_id {sid} does not resolve to any scene!"
                assert sid in scene_events, f"No event timeline found for {sid}"

                gt_timeline = scene_events[sid]
                # Check that ground-truth answer matches QAEngine evaluation on GT timeline
                predicted = engine.answer(q["question"], q["options"], gt_timeline)
                if predicted != q["answer"]:
                    mismatches.append((sid, q["question"], q["answer"], predicted))
                total_qa += 1

    print(f"  [QA INTEGRITY] Verified {total_qa} QA pairs. All scene IDs resolve to real scenes.")
    assert len(mismatches) == 0, f"QA logic mismatches found on ground-truth timelines: {len(mismatches)} (first: {mismatches[:2]})"
    print("  [ORACLE ACCURACY] PASSED: 100.0% consistency between QA ground truth and QAEngine.")

    # 3. Class ordering
    assert len(CLASSES) == N_CLASSES, f"Class count mismatch: {len(CLASSES)} vs {N_CLASSES}"
    print(f"  [ONTOLOGY] PASSED: All {N_CLASSES} classes consistently mapped.")
    print("=== All Dataset Verification Checks Passed Successfully ===")
    return True


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "dataset_mid"
    verify_dataset(target)
