"""Synthetic audio-scene + question/answer dataset generator.

Scenes are 10 s, 16 kHz mono mixtures of isolated ESC-50 sound bursts placed at
known times over a quiet background. Ground-truth event annotations therefore
exist by construction, and questions/answers are produced programmatically from
them. Train/val/test use disjoint ESC-50 folds (1-3 / 4 / 5), so no source
recording appears in more than one split.

Usage:
  python -m acl.data_gen --esc50 /path/to/ESC-50 --out dataset --n_train 1600 --n_val 300 --n_test 500
"""
import argparse
import json
import shutil
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from .classes import (ALERT, BG_CLASSES, CAUSE, CLASS2SCENE, CLASSES, DUR, FG_LABELS,
                      N_SAMPLES, SCENE_LABELS, SCENE_NAMES, SCENES, SR)
from .templates import COUNT_OPTIONS, SUBTYPE_TYPE, YESNO, render

SPLIT_FOLDS = {"train": [1, 2, 3], "val": [4], "test": [5]}
MIN_SAME_CLASS_GAP = 0.6     # s between two events of the same class (keeps counting well-defined)
MAX_EVENT_S = 2.5
MIN_EVENT_S = 0.3


# ----------------------------------------------------------------------------- audio helpers
def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)


def set_rms(x, db):
    return x * 10 ** ((db - rms_db(x)) / 20)


def longest_burst(y, top_db=30):
    """Reduce a clip to its longest non-silent burst => one acoustic event per placed clip."""
    iv = librosa.effects.split(y, top_db=top_db)
    if len(iv) == 0:
        return None
    s, e = max(iv, key=lambda t: t[1] - t[0])
    seg = y[s:min(e, s + int(MAX_EVENT_S * SR))]
    if len(seg) < MIN_EVENT_S * SR:
        return None
    return seg.astype(np.float32)


def build_bank(esc_root, split):
    meta = pd.read_csv(Path(esc_root) / "meta" / "esc50.csv")
    meta = meta[meta.fold.isin(SPLIT_FOLDS[split])]
    fg = {c: [] for c in CLASSES}
    bg = {c: [] for c in BG_CLASSES}
    for r in meta.itertuples():
        if r.category not in fg and r.category not in bg:
            continue
        y, _ = librosa.load(Path(esc_root) / "audio" / r.filename, sr=SR, mono=True)
        if r.category in fg:
            b = longest_burst(y)
            if b is not None:
                fg[r.category].append((b, r.filename))
        else:
            bg[r.category].append((y.astype(np.float32), r.filename))
    for c in CLASSES:
        assert len(fg[c]) >= 5, f"too few usable clips for {c} in {split}"
    return fg, bg


def make_bg(rng, clip):
    y = np.concatenate([clip, clip[::-1]])            # mirror-tile: no discontinuities
    n = N_SAMPLES
    if len(y) < n:
        y = np.tile(y, int(np.ceil(n / len(y))))
    s = int(rng.integers(0, len(y) - n + 1))
    return set_rms(y[s:s + n], float(rng.uniform(-42, -32)))


def fade(x, ms=5):
    n = int(SR * ms / 1000)
    x = x.copy()
    x[:n] *= np.linspace(0, 1, n)
    x[-n:] *= np.linspace(1, 0, n)
    return x


# ----------------------------------------------------------------------------- scene synthesis
def make_scene(rng, fg, bg):
    scene = SCENE_NAMES[int(rng.integers(len(SCENE_NAMES)))]
    pool = SCENES[scene]["fg"]
    while True:
        k = min(len(pool), int(rng.choice([2, 3], p=[0.6, 0.4])))
        classes = [str(c) for c in rng.choice(pool, size=k, replace=False)]
        n_ev = int(rng.integers(max(2, k), 7))               # 2..6 events
        seq = classes + [str(rng.choice(classes)) for _ in range(n_ev - k)]
        rng.shuffle(seq)
        events, wave_parts = [], []
        for cid in seq:
            clip, fname = fg[cid][int(rng.integers(len(fg[cid])))]
            if len(clip) > SR and rng.random() < 0.7:          # random truncation => varied event durations
                clip = clip[:int(len(clip) * float(rng.uniform(0.4, 1.0)))]
            dur = len(clip) / SR
            for _ in range(120):
                on = float(rng.uniform(0.3, DUR - dur - 0.3))
                off = on + dur
                ok, conc = True, 0
                for e in events:
                    ov = min(off, e["offset"]) - max(on, e["onset"])
                    if e["class"] == cid:
                        if ov > -MIN_SAME_CLASS_GAP:
                            ok = False
                            break
                    elif ov > 0:
                        if ov > 0.6 * min(dur, e["offset"] - e["onset"]):
                            ok = False
                            break
                        conc += 1
                if ok and conc <= 1:
                    gain = float(rng.uniform(-28, -16))
                    events.append({"class": cid, "label": FG_LABELS[cid], "onset": round(on, 3),
                                   "offset": round(off, 3), "source": fname, "rms_db": round(gain, 1)})
                    wave_parts.append((on, fade(set_rms(clip, gain))))
                    break
        if len(events) >= 2 and len({e["class"] for e in events}) >= 2:
            break
    bg_cls = SCENES[scene]["bg"]
    bclip, bname = bg[bg_cls][int(rng.integers(len(bg[bg_cls])))]
    y = make_bg(rng, bclip)
    for on, w in wave_parts:
        s = int(round(on * SR))
        y[s:s + len(w)] += w[:N_SAMPLES - s]
    peak = float(np.abs(y).max())
    if peak > 0.98:
        y *= 0.98 / peak
    order = sorted(range(len(events)), key=lambda i: events[i]["onset"])
    events = [events[i] for i in order]
    return y.astype(np.float32), {"scene": scene, "scene_label": SCENE_LABELS[scene],
                                  "background": {"class": bg_cls, "source": bname}, "events": events}


# ----------------------------------------------------------------------------- QA generation
def _opts(rng, correct, others, absent_labels, n=4):
    """Options list of size n that always contains `correct`."""
    others = [o for o in dict.fromkeys(others) if o != correct]
    extra = [a for a in absent_labels if a != correct and a not in others]
    rng.shuffle(extra)
    opts = [correct] + others
    opts = opts[:n]
    for a in extra:
        if len(opts) >= n:
            break
        opts.append(a)
    if correct not in opts:
        opts[-1] = correct
    opts = list(dict.fromkeys(opts))
    rng.shuffle(opts)
    return opts


def gen_qa(rng, rec):
    """Ground-truth QA from annotations. Implemented independently from acl.qa_engine on purpose:
    tests check engine(oracle timeline) == these answers."""
    ev = rec["events"]
    by = defaultdict(list)
    for e in ev:
        by[e["class"]].append(e)
    present = sorted(by)
    absent = [c for c in CLASSES if c not in by]
    lab = FG_LABELS
    first_on = {c: min(e["onset"] for e in by[c]) for c in present}
    qa = []

    def add(sub, options, answer, classes, A=None, B=None):
        assert answer in options, (sub, answer, options)
        qa.append({"type": SUBTYPE_TYPE[sub], "subtype": sub,
                   "question": render(sub, int(rng.integers(2)), A, B),
                   "options": options, "answer": answer, "classes": classes})

    # perceptual: present (multiple choice)
    p = str(rng.choice(present))
    dist = [lab[str(d)] for d in rng.choice(absent, 3, replace=False)]
    o = [lab[p]] + dist
    rng.shuffle(o)
    add("present", o, lab[p], [p])
    # perceptual: scene
    o = list(SCENE_LABELS.values())
    rng.shuffle(o)
    add("scene", o, rec["scene_label"], list(present))
    # perceptual: exists yes / no
    c = str(rng.choice(present))
    add("exists", list(YESNO), "yes", [c], A=c)
    same_scene_absent = [x for x in SCENES[rec["scene"]]["fg"] if x not in by]
    pool = same_scene_absent if (same_scene_absent and rng.random() < 0.5) else absent
    c = str(rng.choice(pool))
    add("exists", list(YESNO), "no", [c], A=c)
    # counting
    c1 = str(rng.choice(present))
    add("count", list(COUNT_OPTIONS), str(len(by[c1])), [c1], A=c1)
    others = [x for x in present if x != c1]
    if others and rng.random() > 0.3:
        c2 = str(rng.choice(others))
    else:
        c2 = str(rng.choice(absent))
    add("count", list(COUNT_OPTIONS), str(len(by.get(c2, []))), [c2], A=c2)
    # temporal: first / order
    pairs = [(a, b) for a, b in combinations(present, 2) if abs(first_on[a] - first_on[b]) >= 0.5]
    if pairs:
        a, b = pairs[int(rng.integers(len(pairs)))]
        if rng.random() < 0.5:
            a, b = b, a
        add("first", [lab[a], lab[b]], lab[a] if first_on[a] < first_on[b] else lab[b], [a, b], A=a, B=b)
        a, b = pairs[int(rng.integers(len(pairs)))]
        if rng.random() < 0.5:
            a, b = b, a
        add("order", list(YESNO), "yes" if first_on[a] < first_on[b] else "no", [a, b], A=a, B=b)
    # temporal: after / before (unique class, no near-simultaneous neighbours)
    singles = [c for c in present if len(by[c]) == 1]
    rng.shuffle(singles)
    for sub in ("after", "before"):
        for c in singles:
            A = by[c][0]
            rest = [e for e in ev if e is not A]
            if any(abs(e["onset"] - A["onset"]) < 0.5 for e in rest):
                continue
            if sub == "after":
                cand = sorted([e for e in rest if e["onset"] > A["onset"]], key=lambda e: e["onset"])
            else:
                cand = sorted([e for e in rest if e["onset"] < A["onset"]], key=lambda e: -e["onset"])
            if len(cand) >= 2 and abs(cand[0]["onset"] - cand[1]["onset"]) < 0.3:
                continue
            ans = lab[cand[0]["class"]] if cand else "nothing"
            o = _opts(rng, ans, [lab[x] for x in present if x != c] + ["nothing"],
                      [lab[x] for x in absent])
            add(sub, o, ans, [c] + ([cand[0]["class"]] if cand else []), A=c)
            break
    # causal + longest
    by_on = sorted(ev, key=lambda e: e["onset"])
    if len(by_on) >= 2 and by_on[1]["onset"] - by_on[0]["onset"] >= 0.5:
        c = by_on[0]["class"]
        o = [CAUSE[c]] + [CAUSE[str(x)] for x in rng.choice([k for k in CLASSES if k != c], 3, replace=False)]
        rng.shuffle(o)
        add("cause_first", o, CAUSE[c], [c])
    by_dur = sorted(ev, key=lambda e: -(e["offset"] - e["onset"]))
    d0 = by_dur[0]["offset"] - by_dur[0]["onset"]
    d1 = by_dur[1]["offset"] - by_dur[1]["onset"]
    if d0 - d1 >= 0.4:
        c = by_dur[0]["class"]
        o = [CAUSE[c]] + [CAUSE[str(x)] for x in rng.choice([k for k in CLASSES if k != c], 3, replace=False)]
        rng.shuffle(o)
        add("cause_longest", o, CAUSE[c], [c])
        o = _opts(rng, lab[c], [lab[x] for x in present], [lab[x] for x in absent])
        add("longest", o, lab[c], [c])
    # overlap
    yes_p, no_p = [], []
    for a, b in combinations(present, 2):
        ovm = max(min(x["offset"], y["offset"]) - max(x["onset"], y["onset"]) for x in by[a] for y in by[b])
        if ovm >= 0.2:
            yes_p.append((a, b))
        elif ovm <= -0.2:
            no_p.append((a, b))
    if yes_p or no_p:
        use_yes = bool(yes_p) and (not no_p or rng.random() < 0.5)
        lst = yes_p if use_yes else no_p
        a, b = lst[int(rng.integers(len(lst)))]
        if rng.random() < 0.5:
            a, b = b, a
        add("overlap", list(YESNO), "yes" if use_yes else "no", [a, b], A=a, B=b)
    # alert (Vibe-style: is there a sound a deaf listener should be notified about?)
    add("alert", list(YESNO), "yes" if any(c in ALERT for c in present) else "no", list(present))
    return qa


# ----------------------------------------------------------------------------- main
def build_split(esc_root, out, split, n, seed):
    rng = np.random.default_rng(seed)
    fg, bg = build_bank(esc_root, split)
    (out / "audio" / split).mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(exist_ok=True)
    (out / "qa").mkdir(exist_ok=True)
    fa = open(out / "annotations" / f"{split}.jsonl", "w")
    fq = open(out / "qa" / f"{split}.jsonl", "w")
    nq = 0
    for i in range(n):
        y, rec = make_scene(rng, fg, bg)
        sid = f"{split}_{i:05d}"
        rel = f"audio/{split}/{sid}.flac"
        sf.write(out / rel, y, SR, format="FLAC", subtype="PCM_16")
        rec.update({"scene_id": sid, "split": split, "file": rel, "duration": DUR})
        fa.write(json.dumps(rec) + "\n")
        for j, q in enumerate(gen_qa(rng, rec)):
            q.update({"qid": f"{sid}_q{j:02d}", "scene_id": sid, "split": split})
            fq.write(json.dumps(q) + "\n")
            nq += 1
    fa.close()
    fq.close()
    print(f"[{split}] {n} scenes, {nq} questions")


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def write_stats_and_card(out, seed):
    stats = {"seed": seed, "splits": {}}
    for split in SPLIT_FOLDS:
        ann = read_jsonl(out / "annotations" / f"{split}.jsonl")
        qa = read_jsonl(out / "qa" / f"{split}.jsonl")
        tc = Counter(q["type"] for q in qa)
        sc = Counter(q["subtype"] for q in qa)
        cnt = Counter(q["answer"] for q in qa if q["subtype"] == "count")
        yn = {s: Counter(q["answer"] for q in qa if q["subtype"] == s) for s in ("exists", "order", "overlap", "alert")}
        stats["splits"][split] = {
            "scenes": len(ann), "questions": len(qa),
            "avg_events_per_scene": round(float(np.mean([len(a["events"]) for a in ann])), 2),
            "scene_types": dict(Counter(a["scene"] for a in ann)),
            "question_types": dict(tc), "question_subtypes": dict(sc),
            "count_answer_dist": dict(sorted(cnt.items())),
            "yes_no_balance": {k: dict(v) for k, v in yn.items()},
        }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    (out / "classes.json").write_text(json.dumps(
        {"foreground_classes": FG_LABELS, "scenes": {k: {"label": v["label"], "background": v["bg"], "classes": v["fg"]}
                                                     for k, v in SCENES.items()}, "cause_table": CAUSE,
         "alert_classes": sorted(ALERT)}, indent=2))
    s = stats["splits"]
    card = f"""# ACL-Synth: Audio Context Layer synthetic QA dataset

Synthetic audio-question-answering dataset for a Proof of Concept "audio context layer".
Every scene is a **10 s, 16 kHz mono FLAC** mixture of isolated sound bursts from
[ESC-50](https://github.com/karolpiczak/ESC-50) placed at known times over a low-level background.
Because scenes are composed programmatically, exact event annotations (class, onset, offset, source clip) exist by
construction, and question/answer pairs are generated from them.

## Splits (no source recording is shared across splits)
| split | ESC-50 folds | scenes | questions |
|---|---|---|---|
| train | 1, 2, 3 | {s['train']['scenes']} | {s['train']['questions']} |
| val | 4 | {s['val']['scenes']} | {s['val']['questions']} |
| test | 5 | {s['test']['scenes']} | {s['test']['questions']} |

## Construction
* {len(CLASSES)} foreground classes in {len(SCENES)} scene types (street, home, farm, celebration); each scene type also has one background class
  (wind, clock tick, rain, crickets) mixed at RMS -42 to -32 dBFS.
* Each ESC-50 clip is reduced to its **longest non-silent burst** (librosa `split`, top_db=30, capped at {MAX_EVENT_S} s, min {MIN_EVENT_S} s; clips longer than 1 s are randomly truncated with p=0.7 to vary durations), so
  one placed clip = one acoustic event. Foreground RMS is drawn from [-28, -16] dBFS (event-to-background SNR roughly 4-26 dB).
* 2-6 events per scene, 2-3 distinct classes. Events of the same class are separated by at least {MIN_SAME_CLASS_GAP} s (counting stays well-defined);
  events of different classes may overlap (at most one concurrent overlap, at most 60 % of the shorter event).
* Questions are generated from annotations with templates (2 paraphrases each). Ambiguous cases are skipped
  (e.g. temporal questions require onset differences >= 0.5 s; overlap questions require overlap >= 0.2 s or a gap >= 0.2 s).

## Question types
| type | subtypes | example |
|---|---|---|
| perceptual | present, scene, exists | "Which sound is present in this audio clip?" |
| counting | count | "How many times does the siren sound occur?" |
| temporal | first, order, after, before | "Which sound is the next one to start after the door knock sound starts?" |
| causal | cause_first, cause_longest | "What is the most likely reason for the first sound in the recording?" |
| other (extra) | longest, overlap, alert | "Does this recording contain a sound that should alert a deaf listener?" |

All questions are closed-set (`options` given; counting uses options 0-5, yes/no questions use `yes`/`no`).

## Files
```
audio/{{train,val,test}}/scene_XXXXX.flac
annotations/{{split}}.jsonl   # one JSON per scene: scene_id, file, scene, scene_label, background, events[{{class,label,onset,offset,source,rms_db}}]
qa/{{split}}.jsonl            # one JSON per question: qid, scene_id, type, subtype, question, options, answer, classes
classes.json                 # class list, scenes, cause table, alert classes
stats.json                   # counts per split / type / subtype
```

## Statistics
Questions per type (train): {s['train']['question_types']}
Questions per type (test): {s['test']['question_types']}
Average events per scene (train): {s['train']['avg_events_per_scene']}
Counting answer distribution (train): {s['train']['count_answer_dist']}

## Known limitations
* Causal and alert labels come from a hand-written knowledge table (one cause per class), not from human annotation.
* Scenes are synthetic (isolated bursts on a stationary background); real recordings are more cluttered.
* Event boundaries are those of the trimmed burst; reverberation/decay tails are not annotated.

## License / attribution
Derived from ESC-50 (Piczak, 2015), whose clips come from Freesound and are distributed under Creative Commons licences;
the ESC-50 dataset is CC BY-NC. This derived dataset is therefore **for non-commercial research/evaluation use** with attribution.
Generation seed: {seed}. Regenerate with `python -m acl.data_gen`.
"""
    (out / "README.md").write_text(card)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esc50", required=True, help="path to the cloned ESC-50 repo")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--n_train", type=int, default=1600)
    ap.add_argument("--n_val", type=int, default=300)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, (split, n) in enumerate([("train", a.n_train), ("val", a.n_val), ("test", a.n_test)]):
        build_split(a.esc50, out, split, n, a.seed * 1000 + i)
    write_stats_and_card(out, a.seed)
    sample = out / "sample"
    sample.mkdir(exist_ok=True)
    for r in read_jsonl(out / "annotations" / "test.jsonl")[:8]:
        shutil.copy(out / r["file"], sample / Path(r["file"]).name)
    print("done ->", out)


if __name__ == "__main__":
    main()
