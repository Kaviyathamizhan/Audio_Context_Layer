"""Answer natural-language questions from an audio-context timeline.

question text --parse--> (subtype, class arguments) --operate on timeline--> answer (one of `options`).
Counting / temporal / duration / overlap questions are deterministic operations on the timeline, so their
errors can be traced to the perception stage (detector), not to reasoning.
"""
from .classes import ALERT, CAUSE, CLASS2SCENE, FG_LABELS, LABEL2CLASS, SCENE_LABELS
from .templates import parse_question

AFTER_BEFORE_MARGIN = 0.25    # s: ignore near-simultaneous onsets (jitter tolerance)
OVERLAP_MIN = 0.1             # s


def _dur(e):
    return e["offset"] - e["onset"]


class QAEngine:
    def answer(self, question, options, timeline):
        sub, args = parse_question(question)
        fn = getattr(self, f"_{sub}", None) if sub else None
        if fn is None:
            return options[0]                                   # unparseable question: uninformed fallback
        ans = fn(args, options, sorted(timeline, key=lambda e: e["onset"]))
        return ans if ans in options else options[0]

    # ---- helpers
    @staticmethod
    def _of(tl, c):
        return [e for e in tl if e["class"] == c]

    # ---- perceptual
    def _present(self, a, options, tl):
        score = {o: sum(_dur(e) * e.get("conf", 1.0) for e in tl if e["class"] == LABEL2CLASS.get(o)) for o in options}
        return max(options, key=lambda o: score[o])

    def _scene(self, a, options, tl):
        s = {}
        for e in tl:
            s[CLASS2SCENE[e["class"]]] = s.get(CLASS2SCENE[e["class"]], 0) + _dur(e) * e.get("conf", 1.0)
        return SCENE_LABELS[max(s, key=s.get)] if s else options[0]

    def _exists(self, a, options, tl):
        return "yes" if self._of(tl, a["A"]) else "no"

    # ---- counting
    def _count(self, a, options, tl):
        return str(min(len(self._of(tl, a["A"])), 5))

    # ---- temporal
    def _first(self, a, options, tl):
        A, B = self._of(tl, a["A"]), self._of(tl, a["B"])
        if A and B:
            return FG_LABELS[a["A"]] if A[0]["onset"] <= B[0]["onset"] else FG_LABELS[a["B"]]
        if A:
            return FG_LABELS[a["A"]]
        if B:
            return FG_LABELS[a["B"]]
        return options[0]

    def _order(self, a, options, tl):
        A, B = self._of(tl, a["A"]), self._of(tl, a["B"])
        if not A:
            return "no"
        return "yes" if (not B or A[0]["onset"] < B[0]["onset"]) else "no"

    def _after(self, a, options, tl):
        A = self._of(tl, a["A"])
        if not A:
            return "nothing" if "nothing" in options else options[0]
        t0 = A[0]["onset"]
        c = [e for e in tl if e["class"] != a["A"] and e["onset"] >= t0 + AFTER_BEFORE_MARGIN]
        return FG_LABELS[min(c, key=lambda e: e["onset"])["class"]] if c else "nothing"

    def _before(self, a, options, tl):
        A = self._of(tl, a["A"])
        if not A:
            return "nothing" if "nothing" in options else options[0]
        t0 = A[0]["onset"]
        c = [e for e in tl if e["class"] != a["A"] and e["onset"] <= t0 - AFTER_BEFORE_MARGIN]
        return FG_LABELS[max(c, key=lambda e: e["onset"])["class"]] if c else "nothing"

    # ---- causal (grounded: the referenced sound is found in the audio, then explained with the cause table)
    def _cause_first(self, a, options, tl):
        return CAUSE[tl[0]["class"]] if tl else options[0]

    def _cause_longest(self, a, options, tl):
        return CAUSE[max(tl, key=_dur)["class"]] if tl else options[0]

    # ---- extra types
    def _longest(self, a, options, tl):
        return FG_LABELS[max(tl, key=_dur)["class"]] if tl else options[0]

    def _overlap(self, a, options, tl):
        A, B = self._of(tl, a["A"]), self._of(tl, a["B"])
        ov = max([min(x["offset"], y["offset"]) - max(x["onset"], y["onset"]) for x in A for y in B], default=-1)
        return "yes" if ov >= OVERLAP_MIN else "no"

    def _alert(self, a, options, tl):
        return "yes" if any(e["class"] in ALERT for e in tl) else "no"
