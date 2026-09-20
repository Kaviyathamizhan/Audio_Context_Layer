"""Frame probabilities -> a structured 'audio context' timeline (list of events)."""
import numpy as np
from scipy.ndimage import median_filter

from .classes import CLASSES, STEP


def decode(probs, thr=0.5, min_len=2, merge_gap=1, median=3):
    """probs: (T, C) in [0,1]. Returns events sorted by onset:
    {'class', 'onset', 'offset', 'conf'}  (times in seconds)."""
    T, C = probs.shape
    events = []
    for c in range(C):
        act = (probs[:, c] >= thr).astype(np.uint8)
        if median > 1:
            act = median_filter(act, size=median, mode="nearest")
        # segments of consecutive active steps
        idx = np.flatnonzero(np.diff(np.concatenate([[0], act, [0]])))
        segs = [[int(s), int(e)] for s, e in zip(idx[::2], idx[1::2])]          # [start, end)
        merged = []
        for s, e in segs:
            if merged and s - merged[-1][1] <= merge_gap:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        for s, e in merged:
            if e - s >= min_len:
                events.append({"class": CLASSES[c], "onset": round(s * STEP, 3), "offset": round(e * STEP, 3),
                               "conf": float(probs[s:e, c].mean())})
    return sorted(events, key=lambda x: (x["onset"], x["class"]))


def context_to_text(timeline, scene=None):
    """Serialise the timeline as text (what an LLM-based reasoner would receive)."""
    lines = [f"- {e['class'].replace('_', ' ')}: {e['onset']:.1f}s to {e['offset']:.1f}s" for e in timeline]
    head = f"Scene guess: {scene}\n" if scene else ""
    return head + "Detected sound events (10 s recording):\n" + ("\n".join(lines) if lines else "- none")
