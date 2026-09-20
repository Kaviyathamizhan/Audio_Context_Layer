"""Evaluation utilities: sound-event detection F1 (segment- and event-based), QA metrics, bootstrap CIs."""
from collections import Counter, defaultdict

import numpy as np

from .classes import CLASSES, DUR

ON_COLLAR = 0.5


def _f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def segment_f1(pred, gt, seg=1.0):
    """Segment-based F1 with 1 s segments. pred/gt: list (per scene) of event lists."""
    tp = fp = fn = 0
    pc = defaultdict(lambda: [0, 0, 0])
    n_seg = int(DUR / seg)
    for P, G in zip(pred, gt):
        for c in CLASSES:
            pa = np.zeros(n_seg, bool)
            ga = np.zeros(n_seg, bool)
            for e in P:
                if e["class"] == c:
                    pa[int(e["onset"] / seg):min(n_seg, int(np.ceil(e["offset"] / seg)))] = True
            for e in G:
                if e["class"] == c:
                    ga[int(e["onset"] / seg):min(n_seg, int(np.ceil(e["offset"] / seg)))] = True
            a, b, d = int((pa & ga).sum()), int((pa & ~ga).sum()), int((~pa & ga).sum())
            tp, fp, fn = tp + a, fp + b, fn + d
            pc[c][0] += a; pc[c][1] += b; pc[c][2] += d
    return _f1(tp, fp, fn)[0], {c: _f1(*v)[0] for c, v in pc.items()}


def match_events(P, G, on_collar=ON_COLLAR):
    """Greedy one-to-one matching on class + onset (+ offset within max(0.5 s, 20 % of duration)).
    Returns (matched pairs [(pi, gi)], unmatched pred idx, unmatched gt idx)."""
    used, pairs = set(), []
    cand = []
    for i, p in enumerate(P):
        for j, g in enumerate(G):
            if p["class"] != g["class"]:
                continue
            d_on = abs(p["onset"] - g["onset"])
            d_off = abs(p["offset"] - g["offset"])
            if d_on <= on_collar and d_off <= max(0.5, 0.2 * (g["offset"] - g["onset"])):
                cand.append((d_on + d_off, i, j))
    used_p, used_g = set(), set()
    for _, i, j in sorted(cand):
        if i not in used_p and j not in used_g:
            used_p.add(i); used_g.add(j); pairs.append((i, j))
    return pairs, [i for i in range(len(P)) if i not in used_p], [j for j in range(len(G)) if j not in used_g]


def event_f1(pred, gt):
    """Event-based F1 (onset collar 0.5 s, offset collar max(0.5 s, 0.2*dur)); micro + per class."""
    tp = fp = fn = 0
    pc = defaultdict(lambda: [0, 0, 0])
    for P, G in zip(pred, gt):
        pairs, up, ug = match_events(P, G)
        tp += len(pairs); fp += len(up); fn += len(ug)
        for i, _ in pairs:
            pc[P[i]["class"]][0] += 1
        for i in up:
            pc[P[i]["class"]][1] += 1
        for j in ug:
            pc[G[j]["class"]][2] += 1
    return _f1(tp, fp, fn)[0], {c: _f1(*pc[c])[0] for c in CLASSES}


# ----------------------------------------------------------------------------- QA
def bootstrap_acc(scene_idx, correct, n_boot=1000, seed=0):
    """Cluster bootstrap over scenes. Returns (mean, lo, hi) of accuracy."""
    scene_idx, correct = np.asarray(scene_idx), np.asarray(correct, float)
    n_sc = scene_idx.max() + 1
    c = np.bincount(scene_idx, weights=correct, minlength=n_sc)
    n = np.bincount(scene_idx, minlength=n_sc).astype(float)
    rng = np.random.default_rng(seed)
    s = rng.integers(0, n_sc, size=(n_boot, n_sc))
    acc = c[s].sum(1) / np.maximum(n[s].sum(1), 1)
    return float(correct.mean()), float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))


class PriorBaseline:
    """Question-only baseline: answer prior per (subtype, class argument) from the TRAIN questions.
    Picks the option that was most often correct in training; ignores the audio entirely."""

    def __init__(self, train_qa):
        from .templates import parse_question
        self.parse = parse_question
        self.cnt = defaultdict(Counter)
        for q in train_qa:
            self.cnt[self._key(q["question"])][q["answer"]] += 1
            self.cnt[(q["subtype"], None)][q["answer"]] += 1

    def _key(self, question):
        sub, a = self.parse(question)
        return (sub, tuple(sorted(a.items())))

    def answer(self, q):
        for key in (self._key(q["question"]), (q["subtype"], None)):
            c = self.cnt.get(key)
            if c:
                return max(q["options"], key=lambda o: c[o])
        return q["options"][0]


def rank_errors(preds_wrong):
    return Counter(preds_wrong).most_common()
