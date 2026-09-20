# Document 03 — App Flow
# Audio Context Layer (ACL)

ACL is an ML/CLI/experiment pipeline rather than a conventional multi-page application. The flow therefore describes inference and experiment journeys.

## Entry Points
1. CLI inference demo: audio clip + natural-language question.
2. Dataset/experiment pipeline.
3. Colab/Kaggle notebook for full-scale GPU execution.

## Inference Journey
```text
Audio + Question
      ↓
Preprocessing
      ↓
Audio Event Detector
      ↓
Frame Probabilities
      ↓
Timeline Decoder
      ↓
Audio Context Layer
      ↓
Question Router
      ↓
Question-Specific Reasoning
      ↓
Grounded Answer + Evidence
```

## Routing
```text
Question
 ├── Perceptual
 ├── Counting
 ├── Temporal
 └── Knowledge-grounded causal interpretation
```

## Perceptual
Identify detected/present events and return the answer.

## Counting
Identify the requested event/count target, count matching ACL events, and return the integer.

## Temporal
Identify referenced events, compare onset/offset relationships, and return the answer.

## Knowledge-Grounded Causal Interpretation
Identify the detected event, use its knowledge mapping, and return a plausible interpretation. This is not genuine causal inference.

## Experiment Journey
```text
ESC-50
 ↓
Synthetic scene generation
 ↓
Ground-truth timeline
 ↓
QA generation
 ↓
Split/leakage verification
 ↓
CRNN + AST/BiGRU training
 ↓
Validation-only decoder tuning
 ↓
Test prediction
 ↓
SED + QA evaluation
 ↓
Error analysis
 ↓
Edge profiling
 ↓
Technical PDF
```

## Error States
- Missing/invalid audio → clear input/preprocessing error.
- Missing checkpoint → report missing artifact.
- Invalid prediction shape/class/frame contract → stop evaluation.
- Missing timeline/QA metadata → stop evaluation.

No authentication, web navigation, or account flow is defined.
