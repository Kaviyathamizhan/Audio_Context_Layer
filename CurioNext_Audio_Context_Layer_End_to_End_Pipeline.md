# CurioNext — Audio Context Layer

## End-to-End PoC Engineering Plan

**Architecture status: LOCKED**

> Goal: build, evaluate, document, and submit a reproducible Audio
> Context Layer PoC.

------------------------------------------------------------------------

## 1. Project Objective

The assignment asks for a proof-of-concept system that takes:

``` text
Audio sample + natural-language question
```

and returns an answer grounded in that audio.

The required question families are:

1.  Perceptual / apparent
2.  Counting
3.  Temporal
4.  Causal / reasoning
5.  Additional useful question types are welcome

The assignment requires an end-to-end PoC containing a structured
dataset, train/validation/test split, reproducible
model/training/evaluation scripts, quantitative test results,
question-type breakdown, basic error analysis, and a technical document
covering problem formulation, research study, dataset, method/design
justification, experimental setup, results, loss curves if trained,
observations, and limitations.

------------------------------------------------------------------------

# 2. LOCKED ARCHITECTURE

``` text
                         ┌──────────────────┐
                         │   AUDIO INPUT    │
                         └────────┬─────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │ Audio Preprocessing      │
                    │ mono / resample / chunk  │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Pretrained Audio Event   │
                    │ Detector / Tagger        │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Temporal Event Builder   │
                    │ event + onset + offset   │
                    │ + confidence             │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ AUDIO CONTEXT LAYER      │
                    │ timeline + counts +      │
                    │ scene/context summary    │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ QUESTION ROUTER          │
                    └────────────┬─────────────┘
                                 │
             ┌───────────────────┼───────────────────┐
             ▼                   ▼                   ▼
        Perceptual           Counting            Temporal
             │                   │                   │
             └───────────────────┼───────────────────┘
                                 ▼
                       Causal / Reasoning
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ ANSWER + EVIDENCE        │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ EVALUATION + ERROR       │
                    │ ANALYSIS                 │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ TECHNICAL REPORT + README│
                    └──────────────────────────┘
```

### Core design decision

The primary system is **not dependent on fine-tuning a 7B audio-language
model**.

The main PoC is:

``` text
Audio
→ event perception
→ timestamped context
→ question-specific reasoning
→ grounded answer
```

A modern audio-language model such as Qwen2-Audio may be used as an
**optional baseline**, but it must not block the core submission.

------------------------------------------------------------------------

# 3. Why This Architecture

A black-box system would look like:

``` text
audio + question → LLM → answer
```

The locked architecture instead exposes:

``` text
audio
→ detected events
→ timeline
→ context
→ reasoning
→ answer
```

This gives us:

- interpretability
- debugging
- explicit counting
- explicit temporal reasoning
- evidence timestamps
- independent perception/reasoning evaluation
- oracle ablation
- easy replacement of individual components

This is especially useful for a PoC because the assignment explicitly
asks for design decisions and justification.

------------------------------------------------------------------------

# 4. End-to-End Data Flow

## Input

``` text
sample.wav
```

## Question

``` text
"What happens after the dog barks?"
```

## Preprocessing

``` text
44.1 kHz stereo
        ↓
16 kHz mono
```

## Event detector

Example output:

``` json
[
  {
    "event": "dog_bark",
    "onset": 1.2,
    "offset": 2.0,
    "confidence": 0.91
  },
  {
    "event": "footsteps",
    "onset": 2.4,
    "offset": 4.8,
    "confidence": 0.87
  }
]
```

## Context layer

``` text
1.2–2.0  dog_bark
2.4–4.8  footsteps
```

## Question router

``` text
"What happens after..."
→ TEMPORAL
```

## Temporal reasoning

``` text
dog_bark ends at 2.0
next event = footsteps
```

## Final answer

``` text
Footsteps occur after the dog barks.
```

## Evidence

``` text
dog_bark: 1.2–2.0 sec
footsteps: 2.4–4.8 sec
```

------------------------------------------------------------------------

# 5. Repository Structure

``` text
audio-context-layer/
│
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── base.yaml
│   ├── dataset.yaml
│   └── model.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── metadata/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_model_exploration.ipynb
│   └── 03_error_analysis.ipynb
│
├── src/
│   └── audio_context/
│       ├── audio/
│       │   ├── loader.py
│       │   ├── preprocessing.py
│       │   └── windowing.py
│       │
│       ├── detection/
│       │   ├── base.py
│       │   ├── pretrained_tagger.py
│       │   ├── postprocess.py
│       │   └── timeline.py
│       │
│       ├── context/
│       │   ├── schema.py
│       │   ├── builder.py
│       │   └── serializer.py
│       │
│       ├── qa/
│       │   ├── router.py
│       │   ├── perceptual.py
│       │   ├── counting.py
│       │   ├── temporal.py
│       │   ├── causal.py
│       │   └── generator.py
│       │
│       ├── models/
│       │   ├── baseline.py
│       │   └── optional_head.py
│       │
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── evaluator.py
│       │   ├── bootstrap.py
│       │   └── error_analysis.py
│       │
│       └── utils/
│           ├── config.py
│           ├── seed.py
│           └── logging.py
│
├── scripts/
│   ├── download_data.py
│   ├── generate_scenes.py
│   ├── generate_qa.py
│   ├── build_dataset.py
│   ├── run_detector.py
│   ├── build_context.py
│   ├── train.py
│   ├── evaluate.py
│   ├── run_baseline.py
│   └── error_analysis.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_timeline.py
│   ├── test_counting.py
│   ├── test_temporal.py
│   ├── test_router.py
│   └── test_dataset_split.py
│
├── artifacts/
│   ├── models/
│   ├── predictions/
│   ├── metrics/
│   ├── plots/
│   └── reports/
│
└── docs/
    ├── technical_report.md
    ├── data_card.md
    └── architecture.md
```

------------------------------------------------------------------------

# 6. Dataset Strategy

## Primary strategy

Build a controlled synthetic audio-scene dataset.

Concept:

``` text
isolated audio clips
       ↓
scene composition
       ↓
known onset/offset metadata
       ↓
automatic QA generation
```

Example:

``` text
0.0–2.0  dog_bark
2.0–5.0  footsteps
5.0–7.0  car_horn
7.0–10.0 rain
```

Because the generator knows the composition, it knows:

- event identity
- event count
- onset
- offset
- ordering
- scene metadata

That makes counting and temporal supervision precise.

A small, intentionally designed dataset is acceptable for this PoC.

------------------------------------------------------------------------

# 7. Dataset Source

A suitable environmental-audio source such as ESC-50 can be evaluated
for the synthetic composition stage.

Before submission, record:

- exact source
- exact version
- license
- classes used
- preprocessing
- whether source clips were reused
- train/validation/test split policy

Do not make unverified licensing claims.

------------------------------------------------------------------------

# 8. Event Ontology

Create one canonical vocabulary.

Example:

``` text
dog_bark
cat_meow
car_horn
siren
door_knock
doorbell
footsteps
clapping
rain
thunder
engine
speech
laughter
crying
alarm
keyboard
phone_ring
applause
glass_break
music
```

The final class list must match the dataset actually created.

Normalize aliases:

``` text
bark
dog barking
dog_bark
```

into:

``` text
dog_bark
```

------------------------------------------------------------------------

# 9. Synthetic Scene Generator

Create:

``` text
scripts/generate_scenes.py
```

Responsibilities:

1.  Select source clips
2.  Normalize them
3.  Trim unnecessary silence
4.  Choose event sequence
5.  Choose onset times
6.  Optionally create repeated events
7.  Optionally create overlaps
8.  Mix the audio
9.  Save the scene
10. Save exact ground-truth metadata

Example metadata:

``` json
{
  "scene_id": "scene_00001",
  "duration": 12.0,
  "events": [
    {
      "event": "dog_bark",
      "onset": 0.8,
      "offset": 1.7
    },
    {
      "event": "dog_bark",
      "onset": 4.2,
      "offset": 5.1
    },
    {
      "event": "footsteps",
      "onset": 2.0,
      "offset": 4.0
    }
  ]
}
```

------------------------------------------------------------------------

# 10. QA Generation

Generate QA directly from metadata.

## Perceptual

``` text
What sounds are present?
Is there a dog barking?
Which animal can be heard?
What is the main sound?
```

## Counting

``` text
How many times does the dog bark?
How many car horns are present?
How many footsteps events occur?
```

## Temporal

``` text
What happens after the dog bark?
What happens before the car horn?
Which occurs first, footsteps or car horn?
What happens last?
```

## Causal

``` text
Why might this suggest a street environment?
Why might this be an emergency-related scene?
What could explain the presence of a siren?
```

## Additional types

``` text
Existence
Overlap
Scene classification
Dominant event
Event ordering
Alert-worthiness
```

------------------------------------------------------------------------

# 11. QA Schema

Recommended JSONL:

``` json
{
  "id": "qa_000001",
  "audio_id": "scene_000001",
  "audio_path": "audio/scene_000001.wav",
  "question": "How many times does the dog bark?",
  "question_type": "counting",
  "answer": "2",
  "answer_type": "integer",
  "evidence": [
    [0.8, 1.7],
    [4.2, 5.1]
  ]
}
```

Scene metadata:

``` json
{
  "scene_id": "scene_000001",
  "audio_path": "audio/scene_000001.wav",
  "duration": 12.0,
  "events": []
}
```

------------------------------------------------------------------------

# 12. Split Strategy

Split at the **scene/source level before QA generation**.

``` text
SCENES
├── train
├── validation
└── test
```

Never allow:

``` text
same scene → train
same scene → test
```

This prevents QA-level leakage.

If the source dataset has predefined folds, preserve them where
appropriate.

------------------------------------------------------------------------

# 13. Dataset Validation

Create a validator.

Check:

- missing audio
- corrupt files
- invalid timestamps
- event onset \>= offset
- empty questions
- empty answers
- invalid labels
- split leakage
- duplicate scene IDs
- duplicate QA IDs
- class imbalance
- question-type imbalance

Run before model development.

------------------------------------------------------------------------

# 14. Audio Preprocessing

Standard internal representation:

``` text
mono
16 kHz
float32
```

Pipeline:

``` text
load
→ convert mono
→ resample
→ amplitude validation/normalization
→ window
→ retain global timestamps
```

Do not destroy temporal alignment.

------------------------------------------------------------------------

# 15. Event Detection

Use a pretrained audio event/tagging model.

Candidate families:

- AST
- PANNs
- CLAP-style zero-shot audio tagging
- another suitable pretrained AudioSet-oriented model

Pick one stable implementation.

Do not waste the deadline comparing many models.

------------------------------------------------------------------------

# 16. Sliding Windows

For longer audio:

``` text
0–2 sec
1–3 sec
2–4 sec
...
```

Keep:

``` text
window_start
window_end
event
confidence
```

Map predictions back to global timestamps.

------------------------------------------------------------------------

# 17. Postprocessing

Raw predictions:

``` text
0–1 dog 0.72
1–2 dog 0.81
2–3 dog 0.19
```

Postprocessing:

``` text
threshold
→ smooth
→ merge adjacent segments
→ remove tiny false positives
```

Output:

``` json
{
  "event": "dog_bark",
  "onset": 0.0,
  "offset": 2.0,
  "confidence": 0.81
}
```

Document the threshold and merge rule.

------------------------------------------------------------------------

# 18. Audio Context Representation

Canonical context:

``` json
{
  "audio_id": "scene_000123",
  "duration": 11.7,
  "events": [
    {
      "event": "car_horn",
      "onset": 1.4,
      "offset": 2.1,
      "confidence": 0.88
    },
    {
      "event": "footsteps",
      "onset": 3.2,
      "offset": 5.0,
      "confidence": 0.79
    }
  ],
  "event_counts": {
    "car_horn": 1,
    "footsteps": 1
  },
  "scene": "street"
}
```

This JSON is the central interface between perception and reasoning.

------------------------------------------------------------------------

# 19. Question Router

Classify the question into:

``` text
PERCEPTUAL
COUNTING
TEMPORAL
CAUSAL
EXISTENCE
OVERLAP
SCENE
```

Examples:

``` text
"What sound is present?"
→ PERCEPTUAL

"How many times does the dog bark?"
→ COUNTING

"What happens after the bell?"
→ TEMPORAL

"Why might this be a street?"
→ CAUSAL

"Is there a siren?"
→ EXISTENCE
```

A deterministic router is acceptable for the controlled PoC.

------------------------------------------------------------------------

# 20. Perceptual Reasoning

Input:

``` text
What sounds are present?
```

Read:

``` text
context.events
```

Return normalized event names.

Example:

``` text
dog_bark
footsteps
car_horn
```

------------------------------------------------------------------------

# 21. Counting Reasoning

Count distinct postprocessed event segments.

Example:

``` text
dog_bark:
1.2–2.0
6.1–6.8
9.4–10.1
```

Answer:

``` text
3
```

Define one count explicitly:

> One occurrence is one distinct event segment after the documented
> temporal postprocessing rule.

Metrics:

- exact match
- within-one accuracy
- MAE

------------------------------------------------------------------------

# 22. Temporal Reasoning

Supported relations:

``` text
before
after
first
last
during
overlap
```

Example:

``` text
0–2 dog_bark
2–5 footsteps
6–7 car_horn
```

Question:

``` text
What happens after the dog bark?
```

Answer:

``` text
footsteps
```

The reasoning engine should operate on timestamps, not guessed textual
patterns.

------------------------------------------------------------------------

# 23. Causal Reasoning

Causal questions are inference.

Do not claim that audio proves causality.

Example:

``` text
Observed:
siren detected

Inference:
could suggest an emergency vehicle nearby
```

Use uncertainty-aware wording:

``` text
might
could suggest
consistent with
```

A controlled knowledge table can be used:

``` json
{
  "siren": ["emergency_vehicle_nearby"],
  "car_horn": ["road_traffic"],
  "doorbell": ["someone_at_door"],
  "rain": ["rainy_outdoor_environment"]
}
```

Clearly label these as inferred/template-derived explanations.

------------------------------------------------------------------------

# 24. Answer Format

Return:

``` json
{
  "answer": "There are two dog barks.",
  "question_type": "counting",
  "confidence": 0.91,
  "evidence": [
    {
      "event": "dog_bark",
      "onset": 1.2,
      "offset": 2.0
    },
    {
      "event": "dog_bark",
      "onset": 6.1,
      "offset": 6.8
    }
  ]
}
```

The evidence field makes answers auditable.

------------------------------------------------------------------------

# 25. Optional LLM Reasoning Layer

For free-form causal questions, a small instruct LLM may receive only
structured context.

Example:

``` text
Audio context:
- dog_bark: 1.2–2.0 sec
- footsteps: 2.4–4.8 sec
- car_horn: 7.1–7.8 sec

Question:
What happens after the footsteps?

Instruction:
Answer only from the supplied context.
```

The LLM should not be allowed to invent unseen acoustic evidence.

------------------------------------------------------------------------

# 26. Baselines

## Baseline A — Question-only

Input:

``` text
question
```

No audio.

Purpose:

``` text
leakage / shortcut detection
```

If this performs unexpectedly well, investigate the dataset.

## Baseline B — Audio-language model

Optional:

``` text
Qwen2-Audio
```

Run only on a manageable test subset.

Purpose:

``` text
direct audio reasoning comparison
```

If debugging exceeds roughly an hour, drop it.

## Baseline C — Oracle context

Use:

``` text
ground-truth event timeline
```

instead of predicted events.

Purpose:

``` text
separate perception errors from reasoning errors
```

------------------------------------------------------------------------

# 27. Oracle Ablation

Compare:

``` text
Oracle context
vs
Predicted context
```

Example interpretation:

``` text
Oracle = 95%
Predicted = 70%
```

The perception/timeline layer is probably the bottleneck.

If:

``` text
Oracle = 75%
Predicted = 70%
```

the reasoning layer is likely the larger problem.

This is one of the strongest error-analysis experiments in the PoC.

------------------------------------------------------------------------

# 28. Evaluation

Evaluate only on the held-out test set.

## Perceptual

Possible:

``` text
accuracy
precision
recall
F1
```

## Counting

``` text
exact match
within-one accuracy
MAE
```

## Temporal

``` text
accuracy
```

for controlled temporal relations.

## Causal

Use controlled/normalized scoring where possible and supplement
open-ended evaluation with manual inspection or an explicitly documented
semantic evaluator.

------------------------------------------------------------------------

# 29. Detector Evaluation

If temporal event detection is evaluated, report:

``` text
precision
recall
F1
```

and, where appropriate:

``` text
confusion matrix
```

Document the temporal matching criterion.

------------------------------------------------------------------------

# 30. Error Taxonomy

Use:

``` text
missed event
false positive
wrong event class
wrong onset/offset
merged repeated events
split single event
counting error
temporal relation error
question-routing error
reasoning error
dataset/label error
ambiguous question
```

Create an error table with examples.

------------------------------------------------------------------------

# 31. Confidence Intervals

Optional but recommended if time permits.

Bootstrap test predictions to obtain:

``` text
metric
95% confidence interval
```

Do not spend submission-critical time on this if the core evaluation is
unfinished.

------------------------------------------------------------------------

# 32. Experiment Tracking

At minimum record:

``` text
experiment_id
dataset_version
model
checkpoint
seed
threshold
window size
metrics
timestamp
hardware
```

Simple JSON/CSV is enough.

W&B is optional.

------------------------------------------------------------------------

# 33. Training

The primary PoC can work without training a large model.

Optional trainable component:

``` text
pretrained audio encoder/tagger
        ↓
small classification head
        ↓
synthetic event classes
```

If training is performed, save:

``` text
train loss
validation loss
epoch
```

and generate loss curves.

Do not fabricate loss curves if no training occurred.

------------------------------------------------------------------------

# 34. Reproducibility

Record:

``` text
Python version
PyTorch version
audio library version
Transformers version
model checkpoint
dataset version
random seed
hardware
configuration
```

Use:

``` text
requirements.txt
```

or:

``` text
pyproject.toml
```

------------------------------------------------------------------------

# 35. Main Scripts

Recommended commands:

``` bash
python scripts/download_data.py
python scripts/generate_scenes.py
python scripts/generate_qa.py
python scripts/build_dataset.py

python scripts/run_detector.py
python scripts/build_context.py

python scripts/train.py
python scripts/evaluate.py
python scripts/run_baseline.py
python scripts/error_analysis.py
```

Training is optional; the command should exist only if training is
actually implemented.

------------------------------------------------------------------------

# 36. One-Command Inference

This must work before submission:

``` bash
python -m audio_context.inference     --audio sample.wav     --question "What happens after the dog barks?"
```

Expected:

``` text
Question:
What happens after the dog barks?

Answer:
Footsteps occur after the dog barks.

Evidence:
dog_bark: 1.2–2.0 sec
footsteps: 2.4–4.8 sec
```

------------------------------------------------------------------------

# 37. Testing

## Unit tests

Test:

``` text
preprocessing
timestamp conversion
event merging
counting
temporal relations
question routing
split validation
```

## Integration test

``` text
audio
→ detector
→ context
→ question
→ answer
```

## Regression test

Keep a small fixed fixture set so future changes can be checked.

------------------------------------------------------------------------

# 38. Optional FastAPI

Only after the ML pipeline works.

Endpoint:

``` text
POST /answer
```

Input:

``` text
audio file
question
```

Output:

``` json
{
  "answer": "...",
  "question_type": "...",
  "evidence": []
}
```

This is optional and should never delay the required evaluation/report.

------------------------------------------------------------------------

# 39. Optional UI

Minimal:

``` text
Upload audio
↓
Enter question
↓
Run
↓
Answer
↓
Show event timeline/evidence
```

Do not spend major time on frontend styling.

------------------------------------------------------------------------

# 40. Technical Report

Follow the assignment terminology.

Recommended structure:

``` text
1. Problem Formulation
2. Research Study
3. Dataset Description
4. Method
5. Design Decisions and Justification
6. Experimental Setup
7. Results
8. Error Analysis
9. Loss Curves (if trained)
10. Observations
11. Limitations
12. Future Work
13. Reproducibility
```

------------------------------------------------------------------------

# 41. Problem Formulation

Define:

``` text
A = audio
Q = natural-language question
E = detected audio events
C = structured audio context
Y = answer
```

Conceptually:

``` text
E = Perception(A)
C = Context(E)
Y = Reason(C, Q)
```

The goal is to produce:

``` text
Y grounded in evidence from A
```

------------------------------------------------------------------------

# 42. Research Study

Investigate and cite relevant work on:

- Audio Question Answering
- Sound Event Detection
- Audio-language models
- Temporal audio reasoning
- Synthetic audio QA
- Audio context modeling

Potential references to investigate:

``` text
Clotho-AQA
MMAU
MMAR
Audio Flamingo
Qwen2-Audio
AudioBench
DCASE audio QA / sound-event tasks
```

For every paper, record:

``` text
problem
dataset
model
method
result
relevance to our PoC
difference from our approach
```

Do not copy benchmark numbers without verifying the original source.

------------------------------------------------------------------------

# 43. Dataset Section

Report:

``` text
source
license
number of source clips
number of generated scenes
number of QA pairs
event classes
question distribution
average duration
split sizes
construction methodology
```

Include the data schema.

------------------------------------------------------------------------

# 44. Method Section

Explain:

``` text
audio preprocessing
→ event detection
→ postprocessing
→ timeline
→ context schema
→ question routing
→ perceptual reasoning
→ counting
→ temporal reasoning
→ causal reasoning
→ answer/evidence
```

Include architecture diagram.

------------------------------------------------------------------------

# 45. Design Decisions Section

For every major decision write:

``` text
Decision
Why it was selected
Alternative considered
Why alternative was not primary
Trade-off
```

Example:

### Explicit context layer

**Decision:** Use timestamped event context as the central
representation.

**Why:** Counting and temporal questions require explicit events and
timestamps.

**Alternative:** End-to-end audio-language model.

**Why not primary:** Higher compute/debugging complexity and less
transparent intermediate evidence for a deadline-constrained PoC.

**Trade-off:** Less flexible for unconstrained questions.

------------------------------------------------------------------------

# 46. Experimental Setup

Document:

``` text
hardware
software
model
dataset
window length
window overlap
threshold
training hyperparameters if applicable
seed
evaluation protocol
```

------------------------------------------------------------------------

# 47. Results

Never invent results.

Use a table like:

| Question Type | Metric                    | Score |
|---------------|---------------------------|------:|
| Perceptual    | Accuracy/F1               |   TBD |
| Counting      | Exact Match               |   TBD |
| Counting      | Within-One                |   TBD |
| Counting      | MAE                       |   TBD |
| Temporal      | Accuracy                  |   TBD |
| Causal        | Controlled/semantic score |   TBD |
| Overall       | Accuracy                  |   TBD |

Replace `TBD` only with measured values.

------------------------------------------------------------------------

# 48. Error Analysis

Include:

- most frequent failure modes
- representative failures
- root cause
- potential fix

Example:

``` text
Failure:
Three repeated dog barks detected as two.

Cause:
Two nearby events were merged during postprocessing.

Fix:
Tune merge threshold or use a stronger event-boundary model.
```

------------------------------------------------------------------------

# 49. Observations

Only report observations supported by experiments.

Possible examples:

``` text
Counting depends strongly on event separation.

Temporal reasoning improves when event boundaries are accurate.

Synthetic generation provides exact labels but may not represent real-world recordings.

The oracle experiment separates perception errors from reasoning errors.
```

------------------------------------------------------------------------

# 50. Limitations

Be explicit:

- Synthetic audio differs from real recordings
- Event overlap can confuse detection
- Short events may be missed
- Causal reasoning is inference, not proof
- Model vocabulary may not perfectly match custom ontology
- Timestamp boundaries can be approximate
- Closed-set QA does not represent arbitrary user questions
- Edge deployment is not fully implemented in the PoC
- Dataset scale is limited

------------------------------------------------------------------------

# 51. Future Work

``` text
1. Fine-tune a stronger audio encoder
2. Improve sound-event localization
3. Integrate a native audio-language model
4. Streaming inference
5. Quantization
6. ONNX/mobile deployment
7. Better causal knowledge grounding
8. Real-world noisy audio evaluation
9. Sensor fusion
10. Personalized context
```

------------------------------------------------------------------------

# 52. Edge Feasibility

Add a short section because an Audio Context Layer can eventually target
constrained devices.

Measure if possible:

``` text
model size
memory
latency
window latency
CPU/GPU requirement
```

Discuss future:

``` text
FP32
→ FP16
→ INT8
→ ONNX/mobile runtime
```

Do not claim edge performance without measuring it.

------------------------------------------------------------------------

# 53. Priority System

## MUST HAVE

``` text
[ ] Dataset
[ ] QA pairs
[ ] Train/validation/test split
[ ] Dataset documentation
[ ] Audio preprocessing
[ ] Working event/context pipeline
[ ] Perceptual QA
[ ] Counting QA
[ ] Temporal QA
[ ] Causal/reasoning QA
[ ] Quantitative test evaluation
[ ] Question-type breakdown
[ ] Error analysis
[ ] Technical report
[ ] README
[ ] Reproducible commands
```

## SHOULD HAVE

``` text
[ ] Oracle ablation
[ ] Question-only baseline
[ ] Detector metrics
[ ] Confusion matrix
[ ] Evidence timestamps
[ ] Unit tests
```

## OPTIONAL

``` text
[ ] Qwen2-Audio baseline
[ ] Trainable detection head
[ ] FastAPI
[ ] UI
[ ] Bootstrap confidence intervals
[ ] Edge latency experiment
[ ] Quantization
```

------------------------------------------------------------------------

# 54. Deadline Strategy

If time becomes tight:

### 3–5 hours remaining

Build:

``` text
~500 scenes
+
pretrained tagger
+
canonical event mapping
+
timeline
+
perceptual
+
counting
+
temporal
+
simple causal templates
+
oracle comparison
+
evaluation
+
report
```

Skip:

``` text
training
audio LLM
FastAPI
UI
complex MLOps
```

### 8–12 hours remaining

Add:

``` text
small trainable detection head
detector evaluation
question-only baseline
audio-LLM baseline if stable
better error analysis
```

------------------------------------------------------------------------

# 55. Git Strategy

Useful commits:

``` text
initialize project
add dataset pipeline
add audio preprocessing
add event detection
add context representation
add QA engine
add evaluation
add baselines
add error analysis
add technical report
final cleanup
```

Never commit:

``` text
.env
credentials
large raw datasets
private keys
temporary outputs
```

------------------------------------------------------------------------

# 56. Definition of Done

The project is done when:

### Inference works

``` bash
python -m audio_context.inference   --audio sample.wav   --question "How many times does the dog bark?"
```

### Evaluation works

``` bash
python scripts/evaluate.py --split test
```

### Repository is reproducible

``` text
README
+
requirements
+
configs
+
scripts
+
source
```

### Report is complete

``` text
problem
research
dataset
method
design justification
setup
results
error analysis
limitations
```

------------------------------------------------------------------------

# 57. Part 2 Preparation

The live round requires:

``` text
clone repository
→ understand unfamiliar code
→ implement/extend a task
→ use AI tools
→ explain reasoning/design decisions
```

Therefore the Part 1 repository must be modular.

Know these files deeply:

``` text
router.py
timeline.py
counting.py
temporal.py
context/schema.py
evaluation/metrics.py
inference.py
```

You should be able to explain:

``` text
Why this architecture?
Why synthetic data?
Why this event detector?
Why this window size?
Why this threshold?
Why these metrics?
Why this split?
What fails?
What would you change?
```

------------------------------------------------------------------------

# 58. Interview Questions to Prepare

### Why an intermediate context layer?

Because counting and temporal reasoning require explicit event and
timestamp information. It separates perception from reasoning and allows
independent evaluation.

### Why synthetic data?

Because exact event identity, count, onset, offset and ordering can be
generated automatically. This provides strong supervision for a small
PoC.

### Why not make Qwen2-Audio the primary system?

Because the objective is a reproducible PoC, not maximum model size. A
structured pipeline is easier to debug, evaluate and defend, while the
audio-language model can still serve as a baseline.

### How do you count repeated events?

After documented temporal postprocessing, each distinct event segment
counts as one occurrence.

### How do you answer "what happens after X"?

Find X's timestamp and select the next valid event in the ordered
timeline.

### Can audio prove causality?

No. Causal answers are contextual inferences and should be expressed as
possibilities rather than direct observations.

### Why oracle evaluation?

It separates perception failures from reasoning failures.

### How do you prevent leakage?

Split at the scene/source level before generating QA pairs.

------------------------------------------------------------------------

# 59. Final Architecture — Locked

``` text
┌────────────────────────────────────────────────────────────┐
│                         AUDIO INPUT                        │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│                    PREPROCESSING                           │
│ mono • resample • normalize • window • timestamp          │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│                 PRETRAINED AUDIO DETECTOR                  │
│             AST / PANNs / suitable tagger                  │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│                 EVENT POSTPROCESSING                       │
│ threshold • smoothing • merge • boundary handling          │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│                  AUDIO CONTEXT LAYER                       │
│ event | onset | offset | confidence | counts | scene       │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│                    QUESTION ROUTER                         │
└───────────────┬────────────┬────────────┬─────────────────┘
                │            │            │
                ▼            ▼            ▼
          PERCEPTUAL     COUNTING     TEMPORAL
                │            │            │
                └────────────┼────────────┘
                             ▼
                    CAUSAL / REASONING
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│                    ANSWER + EVIDENCE                        │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│                  EVALUATION + ABLATIONS                    │
│ overall • per-type • detector • oracle • errors            │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│              TECHNICAL REPORT + README                     │
└────────────────────────────────────────────────────────────┘
```

------------------------------------------------------------------------

# 60. Final Rule

> **Do not optimize for the most impressive model. Optimize for the
> strongest complete PoC that you can explain, reproduce, evaluate and
> defend.**

The target is:

``` text
good problem formulation
        +
good dataset design
        +
sound architecture
        +
working implementation
        +
meaningful evaluation
        +
honest error analysis
        +
clear engineering reasoning
```

That is the complete end-to-end project.
