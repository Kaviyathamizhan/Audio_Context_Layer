"""Unit tests specifically covering the deterministic Audio Context Layer QA Engine.
Verifies perceptual, counting, temporal, causal interpretation, and extra question routing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acl.classes import CAUSE, FG_LABELS, SCENE_LABELS
from acl.qa_engine import QAEngine
from acl.templates import render


def test_qa_engine_perceptual_and_counting():
    eng = QAEngine()
    timeline = [
        {"class": "dog", "onset": 1.0, "offset": 2.5, "conf": 0.9},
        {"class": "dog", "onset": 4.0, "offset": 5.0, "conf": 0.85},
        {"class": "siren", "onset": 3.0, "offset": 7.0, "conf": 0.95},
    ]

    # Counting with exact template rendering
    q_count = render("count", 0, A="dog")  # "How many times does the dog bark sound occur?"
    ans_count = eng.answer(q_count, ["0", "1", "2", "3", "4", "5"], timeline)
    assert ans_count == "2"

    q_count_zero = render("count", 0, A="rooster")
    ans_count_zero = eng.answer(q_count_zero, ["0", "1", "2", "3", "4", "5"], timeline)
    assert ans_count_zero == "0"

    # Exists
    q_exists = render("exists", 0, A="siren")
    ans_exists = eng.answer(q_exists, ["yes", "no"], timeline)
    assert ans_exists == "yes"

    q_exists_no = render("exists", 0, A="fireworks")
    ans_exists_no = eng.answer(q_exists_no, ["yes", "no"], timeline)
    assert ans_exists_no == "no"


def test_qa_engine_temporal_and_causal():
    eng = QAEngine()
    timeline = [
        {"class": "car_horn", "onset": 1.5, "offset": 2.2},
        {"class": "siren", "onset": 4.0, "offset": 7.5},
    ]

    # First event
    q_first = render("first", 0, A="car_horn", B="siren")
    ans_first = eng.answer(
        q_first,
        [FG_LABELS["car_horn"], FG_LABELS["siren"]],
        timeline,
    )
    assert ans_first == FG_LABELS["car_horn"]

    # Order
    q_order = render("order", 0, A="car_horn", B="siren")
    ans_order = eng.answer(q_order, ["yes", "no"], timeline)
    assert ans_order == "yes"

    # Causal interpretation
    q_cause = render("cause_first", 0)
    ans_cause = eng.answer(
        q_cause,
        [CAUSE["car_horn"], CAUSE["siren"]],
        timeline,
    )
    assert ans_cause == CAUSE["car_horn"]


def test_qa_engine_duration_and_alert():
    eng = QAEngine()
    timeline = [
        {"class": "coughing", "onset": 1.0, "offset": 2.0},
        {"class": "glass_breaking", "onset": 3.0, "offset": 4.5},
    ]

    # Longest
    q_long = render("longest", 0)
    ans_long = eng.answer(
        q_long,
        [FG_LABELS["coughing"], FG_LABELS["glass_breaking"]],
        timeline,
    )
    assert ans_long == FG_LABELS["glass_breaking"]

    # Alert (glass_breaking is in ALERT classes)
    q_alert = render("alert", 0)
    ans_alert = eng.answer(q_alert, ["yes", "no"], timeline)
    assert ans_alert == "yes"
