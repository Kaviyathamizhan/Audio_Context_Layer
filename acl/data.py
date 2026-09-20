"""Loading scenes / annotations / QA and building frame-level training targets."""
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .classes import CLASS2IDX, N_CLASSES, N_SAMPLES, STEP, T_STEPS


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def events_to_targets(events):
    """Multi-label activity per 0.1 s step: active if the event covers >= half of the step."""
    t = np.zeros((T_STEPS, N_CLASSES), np.float32)
    for e in events:
        c = CLASS2IDX[e["class"]]
        for k in range(max(0, int(e["onset"] / STEP)), min(T_STEPS, int(e["offset"] / STEP) + 1)):
            ov = min(e["offset"], (k + 1) * STEP) - max(e["onset"], k * STEP)
            if ov >= STEP / 2 - 1e-9:
                t[k, c] = 1.0
    return t


def load_annotations(root, split, limit=None):
    ann = read_jsonl(Path(root) / "annotations" / f"{split}.jsonl")
    return ann[:limit] if limit else ann


def load_waves(root, ann):
    waves = np.zeros((len(ann), N_SAMPLES), np.int16)
    for i, a in enumerate(ann):
        y, _ = sf.read(Path(root) / a["file"], dtype="int16")
        waves[i, :len(y)] = y[:N_SAMPLES]
    return waves


def load_targets(ann):
    return np.stack([events_to_targets(a["events"]) for a in ann])


def load_qa(root, split, scene_ids=None):
    qa = read_jsonl(Path(root) / "qa" / f"{split}.jsonl")
    if scene_ids is not None:
        keep = set(scene_ids)
        qa = [q for q in qa if q["scene_id"] in keep]
    return qa
