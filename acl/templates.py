"""Question templates shared by the generator and the parser.

The QA engine receives only the natural-language question (+ options), parses it
back to (subtype, arguments) with regexes built from these very templates, and
answers from the audio timeline. Unseen phrasings are NOT handled (documented
limitation); an LLM-based parser is the obvious extension.
"""
import re

from .classes import FG_LABELS, LABEL2CLASS

TEMPLATES = {
    "present": ["Which of the following sounds can be heard in this recording?",
                "Which sound is present in this audio clip?"],
    "scene": ["What kind of environment does this recording suggest?",
              "Where was this audio most likely recorded?"],
    "exists": ["Can you hear the {A} sound in this recording?",
               "Is the {A} sound present in this audio?"],
    "count": ["How many times does the {A} sound occur?",
              "How many separate {A} sounds are there in the recording?"],
    "first": ["Which sound occurs first: the {A} sound or the {B} sound?",
              "Between the {A} sound and the {B} sound, which one starts first?"],
    "order": ["Does the {A} sound occur before the {B} sound?",
              "Is the {A} sound heard earlier than the {B} sound?"],
    "after": ["Which sound is the next one to start after the {A} sound starts?",
              "After the {A} sound begins, what is the next sound that begins?"],
    "before": ["Which sound was the last one to start before the {A} sound starts?",
               "What sound begins just before the {A} sound begins?"],
    "cause_first": ["What is the most likely reason for the first sound in the recording?",
                    "Why is the earliest sound in this audio being heard?"],
    "cause_longest": ["What is the most likely reason for the longest sound in the recording?",
                      "Why is the longest-lasting sound in this audio being heard?"],
    "longest": ["Which sound lasts the longest?",
                "Which sound event has the longest duration?"],
    "overlap": ["Do the {A} sound and the {B} sound overlap in time?",
                "Are the {A} sound and the {B} sound heard at the same time?"],
    "alert": ["Does this recording contain a sound that should alert a deaf listener?",
              "Should a deaf person be notified about something in this recording?"],
}

SUBTYPE_TYPE = {
    "present": "perceptual", "scene": "perceptual", "exists": "perceptual",
    "count": "counting",
    "first": "temporal", "order": "temporal", "after": "temporal", "before": "temporal",
    "cause_first": "causal", "cause_longest": "causal",
    "longest": "other", "overlap": "other", "alert": "other",
}
YESNO = ["yes", "no"]
COUNT_OPTIONS = [str(i) for i in range(6)]


def render(subtype, variant, A=None, B=None):
    t = TEMPLATES[subtype][variant % len(TEMPLATES[subtype])]
    if A is not None:
        t = t.replace("{A}", FG_LABELS[A])
    if B is not None:
        t = t.replace("{B}", FG_LABELS[B])
    return t


def _compile():
    alt = "|".join(re.escape(l) for l in sorted(LABEL2CLASS, key=len, reverse=True))
    out = []
    for st, tpls in TEMPLATES.items():
        for t in tpls:
            rx = ""
            for part in re.split(r"(\{A\}|\{B\})", t):
                if part == "{A}":
                    rx += f"(?P<A>{alt})"
                elif part == "{B}":
                    rx += f"(?P<B>{alt})"
                else:
                    rx += re.escape(part)
            out.append((st, re.compile("^" + rx + "$", re.I)))
    return out


_PATTERNS = _compile()


def parse_question(text):
    """-> (subtype, {'A': class_id, 'B': class_id}) or (None, {})"""
    text = text.strip()
    for st, rx in _PATTERNS:
        m = rx.match(text)
        if m:
            args = {k: LABEL2CLASS[v.lower()] for k, v in m.groupdict().items() if v}
            return st, args
    return None, {}
