# Document 02 — TRD
# Audio Context Layer (ACL)

## Architecture
```text
ESC-50 Dataset (2,000 clips)
        ↓
Synthetic Scene Generator
        ↓
Ground-Truth Event Timeline
        ↓
QA Generator

Audio Input
        ↓
Audio Preprocessing
(16 kHz, mono, 10 s)
        ↓
Audio Event Detector
(Main: AST + BiGRU | Baseline: CRNN)
        ↓
Timeline Post-Decoder
(Validation-only tuning)
        ↓
Audio Context Layer
(Onset, offset, class, confidence)
        ↓
Question Router
        ↓
Perceptual / Counting / Temporal / Causal Interpretation
        ↓
Grounded Answer + Evidence
        ↓
Evaluation + Edge Feasibility Benchmark
        ↓
Technical Report
```

## Stack
- Python
- PyTorch
- Transformers
- NumPy
- SciPy
- pandas
- librosa
- soundfile
- pytest
- matplotlib
- reportlab

## Dataset
- ESC-50: 2,000 real recordings, 50 classes.
- Synthetic benchmark: generated 10-second multi-event scenes.
- Source folds: train 1–3, validation 4, test 5.
- Source-level leakage must be zero.

## Models
### Main
AST + BiGRU. AST remains frozen according to the locked implementation; cache AST features where supported.

### Baseline
CRNN.

## ACL
Predicted events contain onset, offset, class, and confidence.

## Decoder
Tune threshold, minimum length, merge gap, and median filtering on validation only. Freeze the selected parameters before test evaluation.

## Question Types
- Perceptual
- Counting
- Temporal
- Knowledge-grounded causal interpretation

The causal component is an interpretation mapping, not genuine causal inference.

## Evaluation
SED: Segment-F1, Event-F1, per-class metrics, missed events, spurious events, boundary/order errors.

QA: per-type and overall accuracy.

Counting: exact, within-1, MAE.

Conditions: predicted ACL context, oracle ground-truth context, train-only question-frequency baseline, and chance only where the answer space is explicitly closed-set.

Uncertainty: cluster bootstrap over test scenes, 95% CIs.

## Repository
```text
Audio ML/
├── acl/
├── tests/
├── scripts/
├── notebooks/
├── configs/
├── dataset/
├── runs/
├── report/
├── requirements.txt
├── README.md
└── .gitignore
```

## Run Artifact Contract
```text
runs/<run_name>/
├── config.json
├── checkpoints/best.pt
├── predictions/probs_val.npy
├── predictions/probs_test.npy
├── metrics/
├── plots/
└── results/
```

Store seed, model, dataset, sample rate, duration, epochs, batch size, learning rate, and relevant model configuration.

## Constraints
Do not change the locked architecture, leak source clips, tune on test, or fabricate results. The full experiment may execute on Google Colab/Kaggle GPU.
