"""Evaluation script: SED metrics, validation-only decode tuning, QA evaluation, error analysis, and figure generation.

Usage:
  python -m acl.evaluate --data dataset --run runs/crnn
  python -m acl.evaluate --data dataset --run runs/ast
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .classes import CLASSES, FG_LABELS, N_CLASSES, T_STEPS
from .data import load_annotations, load_qa
from .metrics import bootstrap_acc, event_f1, segment_f1, PriorBaseline
from .qa_engine import QAEngine
from .timeline import decode

TYPES = ["perceptual", "counting", "temporal", "causal", "other"]


def validate_predictions(probs, split, ann, n_classes=N_CLASSES, t_steps=T_STEPS):
    """Ensure prediction arrays strictly match annotation count, frame resolution, and classes."""
    n_scenes = len(ann)
    assert probs.ndim == 3, f"[{split}] Expected 3D array (N, T, C), got {probs.shape}"
    assert probs.shape[0] == n_scenes, f"[{split}] Scene count mismatch: {probs.shape[0]} probs vs {n_scenes} annotations"
    assert probs.shape[1] == t_steps, f"[{split}] Frame step mismatch: {probs.shape[1]} vs {t_steps}"
    assert probs.shape[2] == n_classes, f"[{split}] Class count mismatch: {probs.shape[2]} vs {n_classes}"
    assert not np.isnan(probs).any(), f"[{split}] Probabilities contain NaN values"
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), f"[{split}] Probabilities outside [0, 1] range"


def tune_decode_on_val(probs_val, gt_val_events, log=print):
    """Grid search decode parameters strictly on validation set to maximize event-F1."""
    thrs = [0.3, 0.4, 0.5, 0.6]
    min_lens = [2, 3, 4]
    merge_gaps = [1, 2]
    medians = [1, 3]

    best_f1, best_params = -1.0, None
    for thr in thrs:
        for min_len in min_lens:
            for merge_gap in merge_gaps:
                for med in medians:
                    decoded = [decode(p, thr=thr, min_len=min_len, merge_gap=merge_gap, median=med) for p in probs_val]
                    f1 = event_f1(decoded, gt_val_events)[0]
                    if f1 > best_f1:
                        best_f1, best_params = f1, {
                            "thr": thr,
                            "min_len": min_len,
                            "merge_gap": merge_gap,
                            "median": med
                        }
    log(f"decode params tuned on val: {best_params} (val event-F1 {best_f1:.3f})")
    return best_params, best_f1


def categorize_error(q, pred_ans, oracle_ans, pred_tl, gt_tl):
    """Trace why a question was answered incorrectly."""
    relevant_classes = q.get("classes", [])
    pred_classes = {e["class"] for e in pred_tl}
    gt_classes = {e["class"] for e in gt_tl}

    for rc in relevant_classes:
        if rc in gt_classes and rc not in pred_classes:
            return "missed_event"
        if rc in pred_classes and rc not in gt_classes:
            return "spurious_event"
    return "boundary_or_order"


def plot_qa_accuracy(by_type, overall_micro, out_path):
    """Figure 1: Test accuracy by question type with error bars."""
    types = [t for t in TYPES if t in by_type] + ["overall"]
    ours = [by_type[t]["ours"] if t != "overall" else overall_micro["ours"] for t in types]
    oracle = [by_type[t]["oracle"] if t != "overall" else overall_micro["oracle"] for t in types]
    prior = [by_type[t]["prior"] if t != "overall" else overall_micro["prior"] for t in types]
    chance = [by_type[t]["chance"] if t != "overall" else overall_micro["chance"] for t in types]

    ci_lo = [by_type[t]["ours_ci"][0] if t != "overall" else overall_micro["ours_ci"][0] for t in types]
    ci_hi = [by_type[t]["ours_ci"][1] if t != "overall" else overall_micro["ours_ci"][1] for t in types]
    yerr = [np.array(ours) - np.array(ci_lo), np.array(ci_hi) - np.array(ours)]

    x = np.arange(len(types))
    w = 0.2

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x - 1.5 * w, ours, w, yerr=yerr, capsize=3, label="Ours (Predicted Context)", color="#2b5c8f")
    ax.bar(x - 0.5 * w, oracle, w, label="Oracle Context", color="#4daf4a")
    ax.bar(x + 0.5 * w, prior, w, label="Question-only Prior", color="#ff7f00")
    ax.bar(x + 1.5 * w, chance, w, label="Chance", color="#999999")

    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in types])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Test Accuracy by Question Family")
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sed_per_class(f1_per_class, out_path):
    """Figure 2: Per-class event-based F1."""
    classes = sorted(f1_per_class, key=lambda c: f1_per_class[c], reverse=True)
    f1s = [f1_per_class[c] for c in classes]
    labels = [FG_LABELS.get(c, c) for c in classes]

    fig, ax = plt.subplots(figsize=(9, 4))
    y = np.arange(len(classes))
    ax.barh(y, f1s, color="#2b5c8f", align="center")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Event-based F1 (test)")
    ax.set_xlim(0, max(max(f1s, default=0.1) * 1.15, 0.2))
    ax.set_title("Per-Class Sound Event Detection Performance")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset", help="path to dataset root")
    ap.add_argument("--run", required=True, help="path to run dir containing config.json and probs_*.npy")
    ap.add_argument("--n_boot", type=int, default=1000, help="bootstrap iterations for confidence intervals")
    ap.add_argument("--seed", type=int, default=42, help="random seed for evaluation baselines and bootstrap")
    ap.add_argument("--out", default=None, help="output directory for results (default: <run>/results)")
    args = ap.parse_args()

    run = Path(args.run)
    out = Path(args.out) if args.out else (run / "results")
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load run metadata & history
    cfg = json.load(open(run / "config.json")) if (run / "config.json").exists() else {}
    hist = json.load(open(run / "history.json")) if (run / "history.json").exists() else {}

    # 2. Load annotations
    ann_val = load_annotations(args.data, "val")
    ann_test = load_annotations(args.data, "test")
    gt_val_events = [a["events"] for a in ann_val]
    gt_test_events = [a["events"] for a in ann_test]

    # 3. Locate and load probability files
    probs_val_path = run / "predictions" / "probs_val.npy"
    if not probs_val_path.exists():
        probs_val_path = run / "probs_val.npy"
    probs_test_path = run / "predictions" / "probs_test.npy"
    if not probs_test_path.exists():
        probs_test_path = run / "probs_test.npy"

    assert probs_val_path.exists(), f"Missing validation probabilities at {probs_val_path}"
    assert probs_test_path.exists(), f"Missing test probabilities at {probs_test_path}"

    probs_val = np.load(probs_val_path).astype(np.float32)
    probs_test = np.load(probs_test_path).astype(np.float32)

    # 4. Contract Validation
    validate_predictions(probs_val, "val", ann_val)
    validate_predictions(probs_test, "test", ann_test)

    # 5. Tune decode parameters strictly on validation
    best_params, val_f1 = tune_decode_on_val(probs_val, gt_val_events)

    # 6. Freeze parameters & decode test set ONCE
    pred_test_events = [decode(p, **best_params) for p in probs_test]

    # 7. Compute Sound Event Detection (SED) test metrics
    test_seg_f1, seg_f1_per_class = segment_f1(pred_test_events, gt_test_events, seg=1.0)
    test_evt_f1, evt_f1_per_class = event_f1(pred_test_events, gt_test_events)
    n_gt_events = sum(len(g) for g in gt_test_events)
    n_pred_events = sum(len(p) for p in pred_test_events)

    print(f"SED test: segment-F1 {test_seg_f1:.3f} | event-F1 {test_evt_f1:.3f} | "
          f"gt events {n_gt_events} | predicted {n_pred_events}")

    # 8. QA Evaluation
    qa_train = load_qa(args.data, "train")
    qa_test = load_qa(args.data, "test")

    prior_model = PriorBaseline(qa_train)
    engine = QAEngine()

    test_ann_map = {a["scene_id"]: i for i, a in enumerate(ann_test)}
    rng = np.random.default_rng(args.seed)

    by_type = defaultdict(list)
    by_subtype = defaultdict(list)
    all_corr_ours, all_corr_oracle, all_corr_prior, all_corr_chance = [], [], [], []
    scene_indices = []

    count_true, count_pred = [], []
    error_causes = Counter()
    error_examples = []
    class_confusions = Counter()

    for q in qa_test:
        sid = q["scene_id"]
        sc_idx = test_ann_map[sid]
        gt_tl = gt_test_events[sc_idx]
        pred_tl = pred_test_events[sc_idx]

        corr_ours = 1.0 if engine.answer(q["question"], q["options"], pred_tl) == q["answer"] else 0.0
        corr_oracle = 1.0 if engine.answer(q["question"], q["options"], gt_tl) == q["answer"] else 0.0
        corr_prior = 1.0 if prior_model.answer(q) == q["answer"] else 0.0
        corr_chance = 1.0 if rng.choice(q["options"]) == q["answer"] else 0.0

        all_corr_ours.append(corr_ours)
        all_corr_oracle.append(corr_oracle)
        all_corr_prior.append(corr_prior)
        all_corr_chance.append(corr_chance)
        scene_indices.append(sc_idx)

        t = q["type"]
        sub = q["subtype"]
        by_type[t].append((sc_idx, corr_ours, corr_oracle, corr_prior, corr_chance))
        by_subtype[sub].append((sc_idx, corr_ours, corr_oracle, corr_prior, corr_chance))

        if sub == "count":
            try:
                count_true.append(int(q["answer"]))
                count_pred.append(int(engine.answer(q["question"], q["options"], pred_tl)))
            except (ValueError, TypeError):
                pass

        if corr_ours == 0.0:
            pred_ans = engine.answer(q["question"], q["options"], pred_tl)
            oracle_ans = engine.answer(q["question"], q["options"], gt_tl)
            err_tag = categorize_error(q, pred_ans, oracle_ans, pred_tl, gt_tl)
            error_causes[err_tag] += 1
            if len(error_examples) < 50:
                error_examples.append({
                    "qid": q.get("qid", ""),
                    "scene_id": sid,
                    "question": q["question"],
                    "options": q["options"],
                    "ground_truth_answer": q["answer"],
                    "predicted_answer": pred_ans,
                    "oracle_answer": oracle_ans,
                    "error_category": err_tag,
                    "predicted_timeline": pred_tl,
                    "ground_truth_timeline": gt_tl
                })

    # Detector confusions
    for p_evs, g_evs in zip(pred_test_events, gt_test_events):
        p_cls = {e["class"] for e in p_evs}
        g_cls = {e["class"] for e in g_evs}
        for gc in g_cls:
            if gc not in p_cls and p_cls:
                for pc in p_cls:
                    class_confusions[(gc, pc)] += 1

    top_confusions = [
        {"gt": k[0], "detected_as": k[1], "count": v}
        for k, v in class_confusions.most_common(10)
    ]

    # Format QA accuracy table
    print("\nTest QA accuracy (ours | oracle | prior | chance):")
    type_res = {}
    for t in TYPES:
        items = by_type.get(t, [])
        if not items:
            continue
        n_q = len(items)
        s_idx = [x[0] for x in items]
        mean_o, lo_o, hi_o = bootstrap_acc(s_idx, [x[1] for x in items], n_boot=args.n_boot, seed=args.seed)
        mean_ora = float(np.mean([x[2] for x in items]))
        mean_pri = float(np.mean([x[3] for x in items]))
        mean_cha = float(np.mean([x[4] for x in items]))
        type_res[t] = {
            "n": n_q,
            "ours": round(mean_o, 3),
            "ours_ci": [round(lo_o, 3), round(hi_o, 3)],
            "oracle": round(mean_ora, 3),
            "prior": round(mean_pri, 3),
            "chance": round(mean_cha, 3)
        }
        print(f"  {t:<11} n={n_q:4d}  {mean_o:.3f} [{lo_o:.3f},{hi_o:.3f}] | {mean_ora:.3f} | {mean_pri:.3f} | {mean_cha:.3f}")

    subtype_res = {}
    for sub, items in by_subtype.items():
        subtype_res[sub] = {
            "n": len(items),
            "ours": round(float(np.mean([x[1] for x in items])), 3),
            "oracle": round(float(np.mean([x[2] for x in items])), 3),
            "prior": round(float(np.mean([x[3] for x in items])), 3),
            "chance": round(float(np.mean([x[4] for x in items])), 3)
        }

    ov_mean, ov_lo, ov_hi = bootstrap_acc(scene_indices, all_corr_ours, n_boot=args.n_boot, seed=args.seed)
    ov_ora = float(np.mean(all_corr_oracle))
    ov_pri = float(np.mean(all_corr_prior))
    ov_cha = float(np.mean(all_corr_chance))
    print(f"  {'overall':<11} n={len(all_corr_ours):4d}  {ov_mean:.3f} | {ov_ora:.3f} | {ov_pri:.3f} | {ov_cha:.3f}")

    overall_micro = {
        "ours": round(ov_mean, 3),
        "ours_ci": [round(ov_lo, 3), round(ov_hi, 3)],
        "oracle": round(ov_ora, 3),
        "prior": round(ov_pri, 3),
        "chance": round(ov_cha, 3)
    }

    macro_ours = float(np.mean([type_res[t]["ours"] for t in type_res]))
    macro_ora = float(np.mean([type_res[t]["oracle"] for t in type_res]))
    macro_pri = float(np.mean([type_res[t]["prior"] for t in type_res]))
    macro_cha = float(np.mean([type_res[t]["chance"] for t in type_res]))

    overall_macro = {
        "ours": round(macro_ours, 3),
        "oracle": round(macro_ora, 3),
        "prior": round(macro_pri, 3),
        "chance": round(macro_cha, 3)
    }

    # Counting metrics
    count_metrics = {"n": 0, "exact": 0.0, "within_1": 0.0, "mae": 0.0}
    if count_true:
        ct = np.array(count_true)
        cp = np.array(count_pred)
        diff = np.abs(ct - cp)
        count_metrics = {
            "exact": round(float((diff == 0).mean()), 3),
            "within_1": round(float((diff <= 1).mean()), 3),
            "mae": round(float(diff.mean()), 3),
            "n": len(ct)
        }
        print(f"counting: {count_metrics}")

    print(f"error causes: {dict(error_causes)}")

    # 9. Format exact schema for make_report.py
    results_data = {
        "run": {
            "backend": cfg.get("backend", "crnn"),
            "trainable_params": cfg.get("params", 301656),
            "batch_size": cfg.get("bs", 32),
            "lr": cfg.get("lr", 0.001),
            "pos_weight": cfg.get("pos_weight", 4.0),
            "epochs_run": len(hist.get("train_loss", [])),
            "best_epoch": hist.get("best_epoch", 1),
            "train_seconds": hist.get("seconds", 0.0),
            "device": cfg.get("device", "cpu")
        },
        "decode_params": best_params,
        "n_questions": len(all_corr_ours),
        "overall_micro": overall_micro,
        "overall_macro_over_types": overall_macro,
        "by_type": type_res,
        "by_subtype": subtype_res,
        "counting_detail": count_metrics,
        "sed": {
            "segment_f1_1s": round(float(test_seg_f1), 3),
            "event_f1": round(float(test_evt_f1), 3),
            "n_pred_events": n_pred_events,
            "n_gt_events": n_gt_events,
            "event_f1_per_class": {c: round(float(v), 3) for c, v in evt_f1_per_class.items()},
            "segment_f1_per_class": {c: round(float(v), 3) for c, v in seg_f1_per_class.items()}
        },
        "errors": {
            "n_wrong": int((np.array(all_corr_ours) == 0).sum()),
            "cause_counts": dict(error_causes),
            "top_class_confusions": top_confusions
        }
    }

    results_file = out / "results.json"
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)

    with open(out / "error_examples.json", "w") as f:
        json.dump(error_examples, f, indent=2)

    # 10. Generate plots expected by make_report.py
    plot_qa_accuracy(type_res, overall_micro, out / "qa_accuracy.png")
    plot_sed_per_class(evt_f1_per_class, out / "sed_per_class.png")

    print(f"results -> {results_file}")
    print(f"figures saved in -> {out}")


if __name__ == "__main__":
    main()
