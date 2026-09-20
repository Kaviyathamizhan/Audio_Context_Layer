"""Build the technical report PDF from results.json / history.json / figures / dataset stats.

  python -m acl.make_report --data dataset --runs runs/ast runs/crnn --author "Your Name" --out report/ACL_technical_report.pdf

The first run in --runs is the main system; the others appear in the comparison table and loss-curve section.
Optional: put extra observations in docs/extra_notes.md (plain paragraphs separated by blank lines) - they are appended
to the Observations section. READ THE PDF AND EDIT the prose before submitting: numbers are auto-filled, judgement is yours.
"""
import argparse
import datetime
import json
import platform
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

TYPES = ["perceptual", "counting", "temporal", "causal", "other"]


def register_fonts():
    d = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("DV", str(d / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DV-B", str(d / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("DV-I", str(d / "DejaVuSans-Oblique.ttf")))
    pdfmetrics.registerFont(TTFont("DV-BI", str(d / "DejaVuSans-BoldOblique.ttf")))
    pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")


def pct(x):
    return f"{100 * x:.1f}%"


def build(args):
    register_fonts()
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontName="DV", fontSize=9, leading=12.6, alignment=TA_JUSTIFY, spaceAfter=5)
    small = ParagraphStyle("small", parent=body, fontSize=7.5, leading=10, alignment=0)
    cell = ParagraphStyle("cell", parent=body, fontSize=7.8, leading=10, alignment=0, spaceAfter=0)
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="DV-B", fontSize=13.5, spaceBefore=10, spaceAfter=5, textColor=colors.HexColor("#1F3A5F"))
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="DV-B", fontSize=10.5, spaceBefore=7, spaceAfter=3, textColor=colors.HexColor("#1F3A5F"))
    title = ParagraphStyle("title", parent=ss["Title"], fontName="DV-B", fontSize=19, leading=23, spaceAfter=4)
    sub = ParagraphStyle("sub", parent=body, fontSize=10, alignment=1, textColor=colors.HexColor("#444444"))
    bullet = ParagraphStyle("bul", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=2)

    S = []
    P = lambda t: S.append(Paragraph(t, body))
    H1 = lambda t: S.append(Paragraph(t, h1))
    H2 = lambda t: S.append(Paragraph(t, h2))
    BL = lambda items: [S.append(Paragraph(i, bullet, bulletText="•")) for i in items]

    def table(rows, widths=None, header=True, font=7.8):
        data = [[Paragraph(str(c), cell) for c in r] for r in rows]
        t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
        st = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#BBBBBB")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
              ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
        if header:
            st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6ECF5")))
        t.setStyle(TableStyle(st))
        S.append(t); S.append(Spacer(1, 6))

    def image(path, width_cm=15.5, caption=None):
        path = Path(path)
        if not path.exists():
            P(f"<i>[missing figure: {escape(str(path))}]</i>"); return
        w, h = ImageReader(str(path)).getSize()
        S.append(Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w))
        if caption:
            S.append(Paragraph(caption, small))
        S.append(Spacer(1, 4))

    runs = [Path(r) for r in args.runs]
    R = [json.load(open(r / "results" / "results.json")) for r in runs]
    H = [json.load(open(r / "history.json")) for r in runs]
    E = [json.load(open(r / "edge.json")) if (r / "edge.json").exists() else None for r in runs]
    main_r, main_h = R[0], H[0]
    stats = json.load(open(Path(args.data) / "stats.json"))
    st = stats["splits"]
    o = main_r["overall_micro"]
    sed = main_r["sed"]
    bt = main_r["by_type"]
    name = {"ast": "Frozen AudioSet-AST + BiGRU head", "crnn": "Log-mel CRNN (from scratch)"}

    # ------------------------------------------------------------------ title
    S.append(Spacer(1, 10))
    S.append(Paragraph("Audio Context Layer: Timeline-Based Audio Question Answering", title))
    S.append(Paragraph("Proof of Concept - technical report", sub))
    S.append(Paragraph(f"{escape(args.author)} &nbsp;|&nbsp; {datetime.date.today().isoformat()}", sub))
    S.append(Spacer(1, 8))
    P("<b>Summary.</b> We build a proof-of-concept <i>audio context layer</i>: a detector converts a 10 s audio clip into a structured "
      "timeline of sound events (class, onset, offset, confidence) and a deterministic reasoner answers perceptual, counting, temporal, "
      "causal and additional (duration, overlap, alert) questions from that timeline. Because no dataset with exact event-level ground "
      f"truth for all these question types was at hand, we curate a synthetic dataset (ACL-Synth: {st['train']['scenes']}/{st['val']['scenes']}/{st['test']['scenes']} "
      f"train/val/test scenes, {st['train']['questions'] + st['val']['questions'] + st['test']['questions']:,} questions). "
      f"On the held-out test split the full pipeline ({name[main_r['run']['backend']]}) answers <b>{pct(o['ours'])}</b> of {main_r['n_questions']:,} questions correctly "
      f"(question-only prior {pct(o['prior'])}, chance {pct(o['chance'])}, ground-truth-timeline upper bound {pct(o['oracle'])}).")

    # ------------------------------------------------------------------ 1 problem
    H1("1. Problem formulation")
    P("Given a 10-second, 16 kHz mono recording <i>a</i> and a natural-language question <i>q</i> with a closed set of answer options <i>O</i>, "
      "predict the correct answer <i>o</i> in <i>O</i>. We cover four required families - <b>perceptual</b> (which sounds are present, what "
      "environment they suggest), <b>counting</b> (how many times an event occurs), <b>temporal</b> (which comes first, what happens before/after "
      "an event) and <b>causal</b> (why a sound is heard) - plus three extra families we consider important for a situational-awareness system: "
      "<b>duration</b> (which sound lasts longest), <b>overlap</b> (do two sounds occur together) and <b>alert</b> (does the recording contain a sound "
      "a deaf listener should be notified about).")
    P("We model the system as two stages, <i>a &#8594; C(a) &#8594; answer</i>. The <b>audio context layer</b> <i>C(a)</i> is an explicit, queryable "
      "representation: a list of events (class, onset, offset, confidence) and a scene estimate. A <b>reasoner</b> <i>R(q, C(a))</i> answers the "
      "question from it. This separation makes counting and temporal answers verifiable by construction, lets us attribute every error to "
      "perception or to reasoning (by replacing <i>C(a)</i> with ground truth), keeps the perception model small enough for on-device use, and "
      "exposes the same representation to a language model (serialised by <font name='DV-I'>context_to_text</font> in the code).")

    # ------------------------------------------------------------------ 2 research
    H1("2. Research study")
    H2("2.1 Audio question answering datasets and benchmarks")
    P("Clotho-AQA (Lipping et al., 2022) provides crowd-sourced binary and single-word questions for Clotho audio clips. MMAU (Sakshi et al., 2024) "
      "contains roughly 10k audio clips with expert-annotated multiple-choice questions over speech, sounds and music, and MMAR (2025) "
      "adds about 1,000 items requiring multi-step reasoning. DCASE 2025 Task 5 (audio question answering) evaluates on bioacoustics, temporal "
      "soundscapes and complex-QA subsets, with Qwen2-Audio-7B, Audio Flamingo 2 and Gemini baselines. These resources are broad and human-curated, but none "
      "offers exact event-level ground truth for counting and ordering questions on controlled scenes, which is what a context layer needs to be tested against; "
      "hence our synthetic dataset.")
    H2("2.2 Large audio-language models")
    P("Qwen2-Audio pairs a Whisper-large-v3 audio encoder with a 7B language model and is instruction-tuned for audio understanding. Audio Flamingo 2 "
      "(Ghosh et al., 2025) uses a CLAP encoder with cross-attention into a small language model and is trained on AudioSkills, a synthetic QA corpus whose "
      "skills include temporal reasoning, counting and contextual sound-event reasoning; Audio Flamingo 3 (2025) extends this line. "
      "These models are strong general audio reasoners, but they are heavy, and temporal reasoning is a reported weak spot (in the DCASE 2025 Task 5 baseline study, arXiv:2505.07365, "
      "Audio Flamingo 2 is comparatively strong on bioacoustics but weak on temporal soundscapes); counting is singled out as a dedicated skill in AudioSkills, which suggests it is not solved by default.")
    H2("2.3 Describe-then-reason and explicit temporal context")
    P("A complementary line of work first converts audio into text and then reasons with a text-only model. TAC (timestamped audio captioning, 2026) reports large gains on "
      "MMAR, MMSU and MMAU-Pro when a timestamped description is passed to a strong text reasoner, and AudioGenie-Reasoner (2025) composes tools and an LLM in a "
      "training-free multi-agent loop. This supports the central design choice here: make time explicit in an intermediate representation instead of hoping an end-to-end model "
      "recovers it implicitly.")
    H2("2.4 Sound event detection and synthetic soundscapes")
    P("AudioSet (Gemmeke et al., 2017) offers weakly-labelled 10 s clips for 527 classes, on which AST (Gong et al., 2021) and PANNs (Kong et al., 2020) are pre-trained and "
      "are widely used as frozen or fine-tuned feature extractors. Frame-level (strong) supervision requires annotated onsets/offsets, which are cheap to obtain when scenes are synthesised "
      "from isolated events - the approach of Scaper (Salamon et al., 2017). Detection quality is conventionally measured with segment-based and event-based F1 (Mesaros et al., 2016), "
      "which we adopt. We implement a small custom mixer instead of Scaper because our scenes need bespoke constraints (minimum same-class gap, ambiguity filters for question generation).")
    H2("2.5 What we take from the literature")
    BL(["Represent time explicitly (timeline) and reason over it separately from perception.",
        "Use synthetic, programmatically annotated scenes to obtain exact answers for counting/temporal questions, while keeping source recordings disjoint between splits.",
        "Report quality per question type with uncertainty, not a single aggregate score, and always include a no-audio baseline.",
        "Use an oracle-timeline ablation to separate perception errors from reasoning errors."])

    # ------------------------------------------------------------------ 3 dataset
    H1("3. Dataset: ACL-Synth")
    P("Every scene is a 10 s, 16 kHz mono mixture of isolated sound bursts from ESC-50 (Piczak, 2015) placed at known times over a stationary background "
      "(wind, clock tick, rain or crickets, depending on the scene type). There are 22 foreground classes grouped into four scene types (street, home, farm, celebration). "
      "Exact annotations (class, onset, offset, source clip, level) exist by construction; questions and answers are generated from them.")
    table([["split", "ESC-50 folds", "scenes", "questions", "avg. events / scene"],
           ["train", "1, 2, 3", st["train"]["scenes"], st["train"]["questions"], st["train"]["avg_events_per_scene"]],
           ["val", "4", st["val"]["scenes"], st["val"]["questions"], st["val"]["avg_events_per_scene"]],
           ["test", "5", st["test"]["scenes"], st["test"]["questions"], st["test"]["avg_events_per_scene"]]],
          widths=[2.2 * cm, 3 * cm, 2.5 * cm, 2.8 * cm, 4 * cm])
    P("<b>Construction rules.</b> (i) Each ESC-50 clip is reduced to its longest non-silent burst (top_db = 30), capped at 2.5 s and randomly truncated to vary durations, so one placed clip is one acoustic event. "
      "(ii) 2-6 events and 2-3 distinct classes per scene; events of the same class are separated by at least 0.6 s so that counting is well-defined, events of different classes may overlap. "
      "(iii) Foreground RMS is drawn from -28 to -16 dBFS and background RMS from -42 to -32 dBFS. (iv) Train/val/test use disjoint ESC-50 folds, so no source recording appears in two splits. "
      "(v) Questions that would be ambiguous are not generated (temporal questions need onset differences of at least 0.5 s; overlap questions need an overlap of at least 0.2 s or a gap of at least 0.2 s).")
    P("<b>Question types.</b> All questions are closed-set with options (counting: 0-5; yes/no; 2- or 4-way multiple choice) and two paraphrases per template.")
    ex = {}
    for q in (json.loads(l) for l in open(Path(args.data) / "qa" / "test.jsonl")):
        ex.setdefault(q["subtype"], q)
    order = ["present", "scene", "exists", "count", "first", "order", "after", "before", "cause_first", "longest", "overlap", "alert"]
    rows = [["type / subtype", "n (test)", "example question", "example answer"]]
    for k in order:
        if k in ex and k in main_r["by_subtype"]:
            rows.append([f"{ex[k]['type']} / {k}", main_r["by_subtype"][k]["n"], escape(ex[k]["question"]), escape(ex[k]["answer"])])
    table(rows, widths=[3.4 * cm, 1.6 * cm, 8.6 * cm, 3.6 * cm])
    P("Full documentation of the data format, construction methodology and statistics ships with the dataset (<font name='DV-I'>README.md</font>, <font name='DV-I'>stats.json</font>, "
      "<font name='DV-I'>classes.json</font>); files are FLAC audio plus JSONL annotations and QA.")

    # ------------------------------------------------------------------ 4 method
    H1("4. Method and design decisions")
    P("<b>Stage 1 - perception (audio to timeline).</b> A frame-level detector outputs, for each 0.1 s step of the 10 s scene, the probability that each of the 22 classes is active "
      "(multi-label, 100 x 22 outputs). We evaluate two interchangeable backends: (a) <b>AST + BiGRU head</b>: an AudioSet-pretrained Audio Spectrogram Transformer is kept <i>frozen</i>; its patch tokens "
      "form a 12 (frequency) x 101 (time) grid, which we average over frequency to obtain a 100-step feature sequence (768-d), cached once per scene; a small head (LayerNorm, linear, BiGRU, linear) is trained on top. "
      "(b) <b>Log-mel CRNN</b>: three conv blocks + BiGRU trained from scratch on 64-band log-mel features (SpecAugment-style masking and random gain), deliberately small (~0.3 M parameters) to gauge what an edge-sized model can do. "
      "Both are trained with binary cross-entropy (positive weight 4, because active cells are only a few percent of all cells) and AdamW with a one-cycle learning-rate schedule; the epoch with the best validation frame-level F1 is kept.")
    P("<b>Timeline decoding.</b> Probabilities are thresholded per class, median-filtered (3 steps), segments closer than a merge gap are joined and very short segments removed. "
      "Threshold, minimum length and merge gap are selected on the validation split (event-based F1); the test split is never used for tuning. The result is the timeline: a list of "
      "(class, onset, offset, confidence). The scene is estimated by confidence-weighted voting of the classes' scene membership.")
    P("<b>Stage 2 - reasoning (timeline to answer).</b> The question text is parsed (regular expressions generated from the same templates as the generator) into a question family and its class arguments; the answer is a deterministic operation on the timeline: "
      "counting = number of separated segments of a class; first / before / after = comparison of onsets (with a 0.25 s tolerance); longest = maximum duration; overlap = temporal intersection above 0.1 s; "
      "causal = locate the referenced event (first or longest) and map its class to a cause via a small knowledge table; alert = whether any detected class is alert-worthy. "
      "The reasoner does not look at audio, so every error can be traced to the timeline.")
    H2("Design decisions and justification")
    table([["decision", "justification"],
           ["Explicit timeline instead of an end-to-end audio LLM", "Counting/temporal answers become exact operations; errors are attributable (oracle ablation); tiny compute budget; timeline can be serialised to text for an LLM later."],
           ["Synthetic scenes with programmatic QA", "Exact ground truth for counting, order and overlap; full control of difficulty; bonus for curating own data; cost: sim-to-real gap (see Limitations)."],
           ["Train/val/test from disjoint ESC-50 folds", "No source recording leaks across splits; scenes only differ in composition, not in memorised clips."],
           ["Single-burst events, &gt;= 0.6 s gap between same-class events", "Makes 'how many times' well defined and prevents label noise from clips that themselves contain several bursts."],
           ["Closed-set answers", "Unambiguous accuracy; no LLM-judge noise; chance level is known per question."],
           ["Frozen AudioSet-pretrained AST + light head", f"Only {st['train']['scenes']} training scenes: pretrained features generalise better than training from scratch; one forward pass per scene, features cached, fully reproducible and fast to train."],
           ["Frequency-averaged AST patch tokens (0.1 s steps)", "Keeps temporal resolution needed for onsets/offsets while reducing 1,212 tokens to 100 x 768 features."],
           ["BiGRU over time", "Adds temporal context and smoothing at negligible cost."],
           ["BCE with positive weight", "Active cells are rare; without weighting, detectors collapse to 'nothing happens'."],
           ["Decoder hyper-parameters tuned on validation", "Prevents test leakage; the decoder matters as much as the network for counting."],
           ["Small CRNN as second backend", "Shows the accuracy/size trade-off and matches on-device constraints (see Section 9)."]],
          widths=[5.2 * cm, 12 * cm])

    # ------------------------------------------------------------------ 5 setup
    H1("5. Experimental setup")
    rr = main_r["run"]
    P(f"<b>Data:</b> {st['train']['scenes']} train / {st['val']['scenes']} val / {st['test']['scenes']} test scenes; the model never sees validation or test scenes during training. "
      f"<b>Main model:</b> {name[rr['backend']]}, {rr['trainable_params']:,} trainable parameters, batch size {rr['batch_size']}, learning rate {rr['lr']}, positive weight {rr['pos_weight']}, "
      f"{rr['epochs_run']} epochs run (best epoch {rr['best_epoch']}), {rr['train_seconds']:.0f} s training on {rr['device']}. Seed {json.load(open(runs[0] / 'config.json'))['seed']}. "
      f"<b>Environment:</b> Python {platform.python_version()}.")
    P("<b>Metrics.</b> <i>QA:</i> accuracy per question type and overall (micro) with 95% cluster-bootstrap confidence intervals (1,000 resamples of test scenes); for counting also exact-match, "
      "within-one accuracy and mean absolute error. <i>Detector:</i> segment-based F1 (1 s segments) and event-based F1 (onset within 0.5 s and offset within max(0.5 s, 20% of duration)), micro-averaged and per class. "
      "<b>Systems compared:</b> <i>ours</i> (detector + reasoner), <i>oracle</i> (ground-truth timeline + the same reasoner: upper bound / perception-vs-reasoning split), <i>prior</i> "
      "(question-only baseline: the option most often correct in training for that question family and class; ignores audio), and <i>chance</i> (uniform guessing).")

    # ------------------------------------------------------------------ 6 results
    H1("6. Results")
    H2("6.1 Question answering (test split)")
    rows = [["question type", "n", "ours (95% CI)", "oracle timeline", "question-only prior", "chance"]]
    for t in TYPES:
        b = bt[t]
        rows.append([t, b["n"], f"{pct(b['ours'])} [{pct(b['ours_ci'][0])}, {pct(b['ours_ci'][1])}]", pct(b["oracle"]), pct(b["prior"]), pct(b["chance"])])
    rows.append(["<b>overall (micro)</b>", main_r["n_questions"], f"<b>{pct(o['ours'])}</b> [{pct(o['ours_ci'][0])}, {pct(o['ours_ci'][1])}]", pct(o["oracle"]), pct(o["prior"]), pct(o["chance"])])
    rows.append(["overall (macro over types)", "", pct(main_r["overall_macro_over_types"]["ours"]), pct(main_r["overall_macro_over_types"]["oracle"]),
                 pct(main_r["overall_macro_over_types"]["prior"]), pct(main_r["overall_macro_over_types"]["chance"])])
    table(rows, widths=[4 * cm, 1.4 * cm, 4.6 * cm, 2.6 * cm, 2.8 * cm, 1.8 * cm])
    image(runs[0] / "results" / "qa_accuracy.png", 14, "Figure 1. Test accuracy by question type. Error bars: 95% cluster-bootstrap CI for the full pipeline.")
    rows = [["subtype", "n", "ours", "oracle", "prior", "chance"]]
    for k, b in main_r["by_subtype"].items():
        rows.append([k, b["n"], pct(b["ours"]), pct(b["oracle"]), pct(b["prior"]), pct(b["chance"])])
    table(rows, widths=[3.6 * cm, 1.6 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm])
    cd = main_r["counting_detail"]
    P(f"<b>Counting</b> ({cd['n']} questions): exact match {pct(cd['exact'])}, within one {pct(cd['within_1'])}, mean absolute error {cd['mae']:.2f}.")
    H2("6.2 Sound event detection (test split)")
    P(f"Segment-based F1 (1 s): <b>{sed['segment_f1_1s']:.3f}</b>; event-based F1: <b>{sed['event_f1']:.3f}</b> "
      f"({sed['n_pred_events']} predicted vs {sed['n_gt_events']} ground-truth events). Decoder parameters selected on validation: {escape(json.dumps(main_r['decode_params']))}.")
    image(runs[0] / "results" / "sed_per_class.png", 13, "Figure 2. Per-class event-based F1 on the test split.")
    if len(R) > 1:
        H2("6.3 Backend comparison")
        rows = [["backend", "trainable params", "seg. F1", "event F1", "QA overall", "counting", "temporal", "causal"]]
        for r_, e_ in zip(R, E):
            rows.append([name[r_["run"]["backend"]], f"{r_['run']['trainable_params']:,}", f"{r_['sed']['segment_f1_1s']:.3f}", f"{r_['sed']['event_f1']:.3f}",
                         pct(r_["overall_micro"]["ours"]), pct(r_["by_type"]["counting"]["ours"]), pct(r_["by_type"]["temporal"]["ours"]), pct(r_["by_type"]["causal"]["ours"])])
        table(rows, widths=[4.6 * cm, 2.4 * cm, 1.5 * cm, 1.6 * cm, 2 * cm, 1.7 * cm, 1.7 * cm, 1.6 * cm])

    # ------------------------------------------------------------------ 7 loss curves
    H1("7. Training curves")
    for r_, h_, p_ in zip(R, H, runs):
        image(p_ / "loss_curves.png", 15, f"Figure. {name[r_['run']['backend']]}: training/validation BCE loss (left) and validation frame-level F1 (right); dashed line = selected epoch ({h_['best_epoch']}).")

    # ------------------------------------------------------------------ 8 observations
    H1("8. Error analysis and observations")
    er = main_r["errors"]
    causes = er["cause_counts"]
    tot = max(er["n_wrong"], 1)
    P(f"<b>Where do errors come from?</b> The full pipeline gets {er['n_wrong']:,} of {main_r['n_questions']:,} test questions wrong. Replacing the detected timeline by the ground-truth timeline "
      f"(same reasoner) gives {pct(o['oracle'])} accuracy, so essentially all errors are perception errors. Classifying each wrong answer by comparing predicted and true events for the classes the question concerns: "
      + ", ".join(f"{escape(k.replace('_', ' '))}: {v} ({pct(v / tot)})" for k, v in sorted(causes.items(), key=lambda kv: -kv[1])) + ".")
    tops = er["top_class_confusions"][:6]
    if tops:
        P("<b>Most frequent detector confusions</b> (ground-truth class &#8594; what was detected instead): " +
          "; ".join(f"{escape(t['gt'])} &#8594; {escape(t['detected_as'])} ({t['count']})" for t in tops) + ".")
    ordered = sorted(TYPES, key=lambda t: bt[t]["ours"])
    P(f"<b>By question type,</b> the pipeline is weakest on {ordered[0]} ({pct(bt[ordered[0]]['ours'])}) and strongest on {ordered[-1]} ({pct(bt[ordered[-1]]['ours'])}). "
      "Counting and temporal answers depend on exact segmentation: a single missed, merged or split event changes the answer, so they are the most sensitive to detector errors, "
      "while presence/scene/alert questions tolerate boundary errors.")
    above = [t for t in TYPES if bt[t]["prior"] - bt[t]["chance"] > 0.10]
    P("<b>No-audio baseline.</b> The question-only prior stays close to chance for most types" +
      (f"; it is noticeably above chance for {', '.join(above)}, which reflects label imbalance in those families (a fixed guess is right more often than uniform guessing), not information about the audio." if above
       else ", confirming that the data cannot be solved from question text alone."))
    P("<b>Example failures.</b> A sample of wrong answers with ground-truth and predicted events is saved in <font name='DV-I'>results/error_examples.json</font>; typical patterns are "
      "quiet events masked by louder overlapping ones (missed events), a long continuous sound broken into two segments (over-counting) and confusions between acoustically similar classes.")
    notes = Path(args.notes)
    if notes.exists() and notes.read_text().strip():
        for para in notes.read_text().strip().split("\n\n"):
            P(escape(para.strip()))

    # ------------------------------------------------------------------ 9 edge
    edge = next((e for e in E if e and e.get("backend") == "crnn"), None)
    ast_edge = next((e for e in E if e and e.get("backend") == "ast"), None)
    if edge or ast_edge:
        H1("9. Edge feasibility")
        if edge:
            txt = (f"The CRNN detector has {edge['trainable_params']:,} parameters ({edge.get('fp32_size_mb', 0):.2f} MB in fp32) and needs "
                   f"{edge.get('cpu_latency_ms_per_10s_fp32_1thread', 0):.0f} ms per 10 s scene on a single CPU thread (real-time factor {edge.get('real_time_factor_fp32', 0):.3f}).")
            if "int8_dynamic_size_mb" in edge:
                txt += (f" Dynamic INT8 quantisation of its linear/GRU layers reduces the model to {edge['int8_dynamic_size_mb']:.2f} MB with "
                        f"{edge.get('cpu_latency_ms_per_10s_int8_1thread', 0):.0f} ms latency; event-F1 changes from {edge.get('event_f1_fp32', 0):.3f} to {edge.get('event_f1_int8_dynamic', 0):.3f}.")
            P(txt + " These are desktop-CPU numbers, not measurements on a wearable SoC; they indicate order of magnitude only.")
        if ast_edge:
            P(f"The AST-based system trains only {ast_edge['trainable_params']:,} parameters, but its frozen backbone has about {ast_edge.get('frozen_ast_backbone_params', 0) / 1e6:.0f} M parameters, "
              "so it is a server-side / upper-bound configuration; for on-device use the backbone would be distilled into a compact student (e.g. the CRNN) using the timeline as target.")

    # ------------------------------------------------------------------ 10 limitations
    H1("10. Limitations and future work")
    BL(["<b>Synthetic-to-real gap.</b> Scenes are isolated bursts on stationary backgrounds without reverberation or moving sources; real recordings are more cluttered. Results measure the pipeline's logic and the detector on controlled data, not real-world robustness.",
        "<b>Closed vocabulary.</b> 22 classes only; unseen sound types are not handled.",
        "<b>Template questions.</b> The reasoner parses a fixed set of templates with regular expressions. Free-form questions need an LLM parser/reasoner over the serialised timeline (supported by <font name='DV-I'>context_to_text</font>, not evaluated here).",
        "<b>Causal labels.</b> Causes come from a one-cause-per-class table and were not human-verified; the causal family therefore mainly tests perception plus lookup, not deeper multi-event causal reasoning.",
        "<b>Counting definition.</b> Count = number of separated bursts; real repeated events (e.g. barking sequences) are harder to delimit.",
        "<b>Statistical scope.</b> Confidence intervals reflect test-scene sampling only; we report a single training seed.",
        "<b>No large audio-language-model baseline.</b> We did not run Qwen2-Audio / Audio Flamingo on ACL-Synth, so we cannot claim the pipeline beats end-to-end models; this is the first extension we would add.",
        "<b>Temporal resolution.</b> 0.1 s steps and 0.5 s evaluation tolerances; sub-100 ms timing is out of scope."])
    H2("Future work")
    BL(["Fine-tune (rather than freeze) the audio encoder, or distil it into the CRNN, and evaluate on real recordings.",
        "Add streaming inference for long audio and sliding-window context; add uncertainty (return 'unsure' when confidence is low).",
        "Replace template parsing with an LLM that reads the timeline; compare with end-to-end audio-language models on the same test set."])

    # ------------------------------------------------------------------ 11 reproducibility
    H1("11. Reproducibility")
    P("Code, dataset generator and report generator are in the accompanying repository. Reproduction: (1) <font name='DV-I'>python -m acl.data_gen --esc50 ESC-50 --out dataset</font>; "
      "(2) <font name='DV-I'>python -m acl.train_sed --data dataset --backend ast --out runs/ast</font> (and <font name='DV-I'>--backend crnn</font>); "
      "(3) <font name='DV-I'>python -m acl.evaluate --data dataset --run runs/ast</font>; (4) <font name='DV-I'>python -m acl.edge_report</font> and <font name='DV-I'>python -m acl.make_report</font>. "
      "All random seeds are fixed; the dataset is regenerated bit-for-bit from the seed given in <font name='DV-I'>stats.json</font> (up to library versions).")

    H1("References")
    refs = [
        "Chu, Y. et al. (2024). Qwen2-Audio technical report. arXiv:2407.10759.",
        "Gemmeke, J. et al. (2017). Audio Set: An ontology and human-labeled dataset for audio events. ICASSP.",
        "Ghosh, S. et al. (2025). Audio Flamingo 2: An audio-language model with long-audio understanding and expert reasoning abilities. arXiv:2503.03983.",
        "Audio Flamingo 3: Advancing audio intelligence with fully open large audio language models (2025). arXiv:2507.08128.",
        "Gong, Y., Chung, Y.-A., Glass, J. (2021). AST: Audio Spectrogram Transformer. Interspeech.",
        "Kong, Q. et al. (2020). PANNs: Large-scale pretrained audio neural networks for audio pattern recognition. IEEE/ACM TASLP.",
        "Lipping, S., Sudarsanam, P., Drossos, K., Virtanen, T. (2022). Clotho-AQA: A crowdsourced dataset for audio question answering. EUSIPCO.",
        "MMAR: A challenging benchmark for deep reasoning in speech, audio, music, and their mix (2025).",
        "Mesaros, A., Heittola, T., Virtanen, T. (2016). Metrics for polyphonic sound event detection. Applied Sciences, 6(6).",
        "Piczak, K. J. (2015). ESC: Dataset for environmental sound classification. ACM Multimedia.",
        "Sakshi, S. et al. (2024). MMAU: A massive multi-task audio understanding and reasoning benchmark. arXiv:2410.19168.",
        "Salamon, J. et al. (2017). Scaper: A library for soundscape synthesis and augmentation. WASPAA.",
        "Multi-domain audio question answering benchmark toward acoustic content reasoning (DCASE 2025 Task 5 baseline study) (2025). arXiv:2505.07365.",
        "TAC: Timestamped audio captioning (2026). arXiv:2602.15766.",
        "AudioGenie-Reasoner: A training-free multi-agent framework for coarse-to-fine audio deep reasoning (2025). arXiv:2509.16971.",
    ]
    for r_ in refs:
        S.append(Paragraph(escape(r_), small))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.9 * cm, rightMargin=1.9 * cm, topMargin=1.7 * cm, bottomMargin=1.7 * cm,
                            title="Audio Context Layer - technical report", author=args.author)
    doc.build(S, onFirstPage=_footer, onLaterPages=_footer)
    print("report ->", out, f"({out.stat().st_size / 1e6:.2f} MB)")


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 7)
    canvas.setFillColor(colors.grey)
    canvas.drawRightString(A4[0] - 1.9 * cm, 1.0 * cm, f"Audio Context Layer PoC - page {doc.page}")
    canvas.restoreState()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--runs", nargs="+", default=["runs/ast"])
    ap.add_argument("--author", default="Your Name")
    ap.add_argument("--notes", default="docs/extra_notes.md")
    ap.add_argument("--out", default="report/ACL_technical_report.pdf")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
