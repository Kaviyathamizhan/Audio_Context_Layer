"""Unit tests for SED and QA metrics."""
import numpy as np
import pytest

from acl.metrics import segment_f1, event_f1, bootstrap_acc, PriorBaseline


def test_segment_f1_perfect_and_empty():
    gt = [[{"class": "dog", "onset": 1.0, "offset": 2.0}]]
    pred = [[{"class": "dog", "onset": 1.0, "offset": 2.0}]]
    f1, _ = segment_f1(pred, gt, seg=1.0)
    assert f1 == 1.0

    empty = [[]]
    f1_empty, _ = segment_f1(empty, gt, seg=1.0)
    assert f1_empty == 0.0


def test_event_f1_collar_tolerance():
    gt = [[{"class": "dog", "onset": 1.0, "offset": 2.0}]]
    # Within collar (0.5s onset tolerance, offset tolerance)
    pred_near = [[{"class": "dog", "onset": 1.2, "offset": 2.1}]]
    f1_near, _ = event_f1(pred_near, gt)
    assert f1_near == 1.0

    # Outside collar
    pred_far = [[{"class": "dog", "onset": 2.0, "offset": 3.0}]]
    f1_far, _ = event_f1(pred_far, gt)
    assert f1_far == 0.0


def test_bootstrap_acc_bounds():
    scene_idx = [0, 0, 1, 1, 2, 2]
    correct = [1, 1, 1, 0, 0, 0]
    mean, lo, hi = bootstrap_acc(scene_idx, correct, n_boot=200, seed=42)
    assert 0.0 <= lo <= mean <= hi <= 1.0
    assert abs(mean - 0.5) < 1e-5


def test_prior_baseline_train_only():
    train_qa = [
        {"question": "Is the dog sound present?", "subtype": "exists", "options": ["yes", "no"], "answer": "yes"},
        {"question": "Is the dog sound present?", "subtype": "exists", "options": ["yes", "no"], "answer": "yes"},
        {"question": "Is the dog sound present?", "subtype": "exists", "options": ["yes", "no"], "answer": "no"},
    ]
    prior = PriorBaseline(train_qa)
    q = {"question": "Is the dog sound present?", "subtype": "exists", "options": ["yes", "no"]}
    assert prior.answer(q) == "yes"


if __name__ == "__main__":
    test_segment_f1_perfect_and_empty()
    test_event_f1_collar_tolerance()
    test_bootstrap_acc_bounds()
    test_prior_baseline_train_only()
    print("PASS: Metrics tests verified.")
