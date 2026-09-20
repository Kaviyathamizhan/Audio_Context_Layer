"""Sanity tests. Run:  ESC50_DIR=/path/to/ESC-50 python -m pytest -q tests   (or: python tests/test_pipeline.py)

The key test: answers computed by the QA engine from the GROUND-TRUTH timeline must equal the answers written by the
independent generator code for every generated question (100 % oracle accuracy).
"""
import itertools
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acl.classes import CLASSES  # noqa: E402
from acl.data import events_to_targets, load_annotations, load_qa  # noqa: E402
from acl.metrics import event_f1, segment_f1  # noqa: E402
from acl.qa_engine import QAEngine  # noqa: E402
from acl.templates import TEMPLATES, parse_question, render  # noqa: E402
from acl.timeline import decode  # noqa: E402

ESC50 = os.environ.get("ESC50_DIR", "ESC-50").strip()
if not Path(ESC50).exists():
    ESC50 = None


def test_parser_roundtrip():
    for st, tpls in TEMPLATES.items():
        for v in range(len(tpls)):
            for a, b in itertools.product(CLASSES[:5], CLASSES[5:8]):
                sub, args = parse_question(render(st, v, a, b))
                assert sub == st
                if "{A}" in tpls[v]:
                    assert args["A"] == a
                if "{B}" in tpls[v]:
                    assert args["B"] == b


def test_metrics_perfect():
    ev = [[{"class": "dog", "onset": 1.0, "offset": 2.0}, {"class": "siren", "onset": 4.0, "offset": 6.0}]]
    assert event_f1(ev, ev)[0] == 1.0 and segment_f1(ev, ev)[0] == 1.0


def _make_data(tmp):
    from acl.data_gen import build_split
    build_split(ESC50, Path(tmp), "test", 25, seed=123)
    return load_annotations(tmp, "test"), load_qa(tmp, "test")


def test_oracle_engine_matches_generator():
    if not ESC50:
        print("skip (set ESC50_DIR)"); return
    with tempfile.TemporaryDirectory() as tmp:
        ann, qa = _make_data(tmp)
        by = {a["scene_id"]: a["events"] for a in ann}
        eng, bad = QAEngine(), []
        for q in qa:
            pred = eng.answer(q["question"], q["options"], by[q["scene_id"]])
            if pred != q["answer"]:
                bad.append((q["question"], q["answer"], pred))
        assert not bad, bad[:5]
        assert len(qa) > 200


def test_decode_recovers_ground_truth_from_perfect_probs():
    if not ESC50:
        print("skip (set ESC50_DIR)"); return
    with tempfile.TemporaryDirectory() as tmp:
        ann, _ = _make_data(tmp)
        gt = [a["events"] for a in ann]
        pred = [decode(events_to_targets(e), thr=0.5, min_len=2, merge_gap=1) for e in gt]
        assert event_f1(pred, gt)[0] >= 0.97
        assert segment_f1(pred, gt)[0] >= 0.97


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
