"""Train the frame-level sound event detector and save probabilities for val/test.

  python -m acl.train_sed --data dataset --backend ast  --out runs/ast   # frozen AudioSet-pretrained AST + trainable head
  python -m acl.train_sed --data dataset --backend crnn --out runs/crnn  # small CRNN from scratch (edge-sized)

Outputs in --out: best.pt, history.json (loss curves), config.json, loss_curves.png, probs_{val,test}.npy
(+ cached AST features feats_{split}.npy).
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .classes import N_CLASSES
from .data import load_annotations, load_targets, load_waves
from .models import build_model, extract_ast_features, n_params


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def frame_f1(logits, y, thr=0.5):
    p = (torch.sigmoid(logits) >= thr)
    y = y >= 0.5
    tp = (p & y).sum().item(); fp = (p & ~y).sum().item(); fn = (~p & y).sum().item()
    return 2 * tp / max(2 * tp + fp + fn, 1)


def get_inputs(args, split, ann, out, device):
    """CRNN: int16 waveforms. AST: cached frozen features."""
    if args.backend == "crnn":
        return torch.from_numpy(load_waves(args.data, ann))
    cache = out / f"feats_{split}.npy"
    if cache.exists() and np.load(cache, mmap_mode="r").shape[0] == len(ann):
        return torch.from_numpy(np.load(cache))
    waves = load_waves(args.data, ann)
    print(f"[{split}] extracting frozen AST features ({args.ast_model}) ...", flush=True)
    f = extract_ast_features(waves, args.ast_model, device=device, batch=args.ast_batch,
                             random_init=args.ast_random_init)
    np.save(cache, f)
    return torch.from_numpy(f)


def prep(x, backend, device, train=False):
    x = x.to(device)
    if backend == "crnn":
        x = x.float() / 32768.0
        if train:                                             # random gain +-6 dB
            g = 10 ** (torch.empty(x.shape[0], 1, device=device).uniform_(-6, 6) / 20)
            x = x * g
        return x
    return x.float()


@torch.no_grad()
def predict(model, X, backend, device, bs=64):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(torch.sigmoid(model(prep(X[i:i + bs], backend, device))).cpu())
    return torch.cat(out).numpy()


def plot_curves(hist, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = range(1, len(hist["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(ep, hist["train_loss"], label="train"); ax[0].plot(ep, hist["val_loss"], label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("BCE loss"); ax[0].set_title("Loss curves"); ax[0].legend()
    ax[1].plot(ep, hist["val_f1"], color="tab:green")
    ax[1].axvline(hist["best_epoch"], ls="--", color="grey", lw=1)
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val frame F1 @0.5"); ax[1].set_title("Validation frame-level F1")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--backend", choices=["crnn", "ast"], default="ast")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--pos_weight", type=float, default=4.0)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="use only N scenes per split (smoke tests)")
    ap.add_argument("--ast_model", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    ap.add_argument("--ast_batch", type=int, default=16)
    ap.add_argument("--ast_random_init", action="store_true", help=argparse.SUPPRESS)  # offline unit test only
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.epochs = args.epochs or (40 if args.backend == "crnn" else 30)
    args.lr = args.lr or (2e-3 if args.backend == "crnn" else 1e-3)
    out = Path(args.out or f"runs/{args.backend}")
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| backend:", args.backend, flush=True)

    ann = {s: load_annotations(args.data, s, args.limit) for s in ("train", "val", "test")}
    Y = {s: torch.from_numpy(load_targets(ann[s])) for s in ann}
    X = {s: get_inputs(args, s, ann[s], out, device) for s in ann}
    in_dim = X["train"].shape[-1] if args.backend == "ast" else 0
    model = build_model(args.backend, in_dim=in_dim).to(device)
    print(f"trainable parameters: {n_params(model):,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * int(np.ceil(len(X["train"]) / args.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.15)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.full((N_CLASSES,), args.pos_weight, device=device))

    hist = {"train_loss": [], "val_loss": [], "val_f1": [], "best_epoch": 0, "seconds": 0}
    best, bad, t0 = -1, 0, time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(X["train"]))
        tl, n = 0.0, 0
        for i in range(0, len(perm), args.bs):
            idx = perm[i:i + args.bs]
            x, y = prep(X["train"][idx], args.backend, device, train=True), Y["train"][idx].to(device)
            loss = lossf(model(x), y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            tl += loss.item() * len(idx); n += len(idx)
        model.eval()
        with torch.no_grad():
            vl, vlog = 0.0, []
            for i in range(0, len(X["val"]), 64):
                lg = model(prep(X["val"][i:i + 64], args.backend, device))
                vl += lossf(lg, Y["val"][i:i + 64].to(device)).item() * lg.shape[0]
                vlog.append(lg.cpu())
            vlog = torch.cat(vlog)
        vf1 = frame_f1(vlog, Y["val"])
        hist["train_loss"].append(tl / n); hist["val_loss"].append(vl / len(X["val"])); hist["val_f1"].append(vf1)
        print(f"epoch {ep:3d} | train {tl / n:.4f} | val {vl / len(X['val']):.4f} | val frame-F1 {vf1:.3f}", flush=True)
        if vf1 > best:
            best, bad, hist["best_epoch"] = vf1, 0, ep
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping"); break
    hist["seconds"] = round(time.time() - t0, 1)
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))

    # Save to both run root and structured subdirectories
    (out / "checkpoints").mkdir(exist_ok=True)
    (out / "predictions").mkdir(exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    (out / "metrics").mkdir(exist_ok=True)
    torch.save(model.state_dict(), out / "checkpoints" / "best.pt")

    for s in ("val", "test"):
        preds = predict(model, X[s], args.backend, device).astype(np.float16)
        np.save(out / f"probs_{s}.npy", preds)
        np.save(out / "predictions" / f"probs_{s}.npy", preds)

    json.dump(hist, open(out / "history.json", "w"), indent=1)
    json.dump(hist, open(out / "metrics" / "history.json", "w"), indent=1)

    cfg_data = {**vars(args), "params": n_params(model), "device": device, "in_dim": in_dim,
                "n_train_scenes": len(X["train"])}
    json.dump(cfg_data, open(out / "config.json", "w"), indent=1)

    plot_curves(hist, out / "loss_curves.png")
    plot_curves(hist, out / "plots" / "loss_curves.png")
    print(f"done. best epoch {hist['best_epoch']} (val frame-F1 {best:.3f}); {hist['seconds']} s -> {out}")


if __name__ == "__main__":
    main()
