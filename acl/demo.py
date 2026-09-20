"""Inference demo: build the audio context (timeline) for a wav file and answer questions about it.

  python -m acl.demo --run runs/ast --wav dataset/sample/test_00000.flac
  python -m acl.demo --run runs/ast --wav clip.wav --question "How many times does the siren sound occur?" --options 0 1 2 3 4 5
"""
import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import torch

from .classes import CAUSE, FG_LABELS, N_SAMPLES, SCENE_LABELS, SR
from .models import build_model, extract_ast_features
from .qa_engine import QAEngine
from .templates import COUNT_OPTIONS, YESNO
from .timeline import context_to_text, decode


def load_wave(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    y = np.pad(y, (0, max(0, N_SAMPLES - len(y))))[:N_SAMPLES]
    return (np.clip(y, -1, 1) * 32767).astype(np.int16)


def build_timeline(run, wav_path, device="cpu"):
    run = Path(run)
    cfg = json.load(open(run / "config.json"))
    dp = json.load(open(run / "results" / "results.json"))["decode_params"]
    w = load_wave(wav_path)[None]
    model = build_model(cfg["backend"], in_dim=cfg["in_dim"])
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    model.to(device).eval()
    with torch.no_grad():
        if cfg["backend"] == "crnn":
            probs = torch.sigmoid(model(torch.from_numpy(w).float().to(device) / 32768.0))[0].cpu().numpy()
        else:
            f = extract_ast_features(w, cfg["ast_model"], device=device, batch=1, log=lambda *_: None)
            probs = torch.sigmoid(model(torch.from_numpy(f).float().to(device)))[0].cpu().numpy()
    return decode(probs, **dp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--question", default=None)
    ap.add_argument("--options", nargs="*", default=None)
    a = ap.parse_args()
    tl = build_timeline(a.run, a.wav)
    eng = QAEngine()
    scene = eng.answer("What kind of environment does this recording suggest?", list(SCENE_LABELS.values()), tl)
    print(context_to_text(tl, scene))
    if a.question:
        opts = a.options or YESNO
        print(f"\nQ: {a.question}\nA: {eng.answer(a.question, opts, tl)}")
        return
    print("\nAutomatic questions:")
    seen = sorted({e["class"] for e in tl})
    for c in seen:
        q = f"How many times does the {FG_LABELS[c]} sound occur?"
        print(f"  Q: {q}  A: {eng.answer(q, COUNT_OPTIONS, tl)}")
    if tl:
        lab_opts = list(dict.fromkeys(FG_LABELS[e["class"]] for e in tl))
        cause_opts = list(dict.fromkeys(CAUSE[e["class"]] for e in tl))
        for q, opts in [("What is the most likely reason for the first sound in the recording?", cause_opts),
                        ("Which sound lasts the longest?", lab_opts)]:
            print(f"  Q: {q}  A: {eng.answer(q, opts, tl)}")
    q = "Does this recording contain a sound that should alert a deaf listener?"
    print(f"  Q: {q}  A: {eng.answer(q, YESNO, tl)}")


if __name__ == "__main__":
    main()
