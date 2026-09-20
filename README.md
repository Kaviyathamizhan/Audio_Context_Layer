# Audio Context Layer (ACL)

[![Tests](https://img.shields.io/badge/pytest-10%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](requirements.txt)

An explainable audio understanding system that separates acoustic sound event detection from deterministic contextual reasoning over explicit event timelines.

---

## 1. Overview

Conventional audio event detection (SED) systems output isolated frame probabilities or classification tags, lacking the temporal structure required for compositional reasoning. Conversely, end-to-end audio-language models frequently struggle with exact event counting, precise temporal localization, and boundary-sensitive reasoning.

The **Audio Context Layer (ACL)** bridges this gap by decoupling perception from reasoning. A neural detector first transcribes a 10-second audio clip into a structured, queryable **Audio Context Layer** (a timeline of sound events with class labels, onset timestamps, offset timestamps, and detection confidences). A deterministic question router then answers natural-language queries directly from this structured context.

The system supports queries covering:
- **Perceptual**: What sound events are present; what environment or scene they indicate.
- **Counting**: How many times a specific sound event occurs.
- **Temporal**: Which event occurred first; what happened before or after an event; event ordering.
- **Duration & Overlap**: Which event lasted the longest; whether two events overlapped in time.
- **Alert / Safety**: Whether any sound warrants immediate notification for a deaf or hard-of-hearing listener.
- **Knowledge-Grounded Causal Interpretation**: Plausible interpretations of why an event occurred, derived from an ontology lookup table.

> **Important**: ACL performs **knowledge-grounded causal interpretation** via structured class-to-cause mappings. It does **not** perform genuine mathematical or counterfactual causal inference.

---

## 2. Architecture

```text
ESC-50 Real Recordings (Folds 1–5)
              ↓
  Synthetic Scene Generator
              ↓
  Ground-Truth Event Timelines
              ↓
        QA Generator
              ↓
  Audio Preprocessing (16 kHz, mono, 10.0 s)
              ↓
    Sound Event Detection (SED)
     ├── CRNN Baseline (Log-mel CNN3 + BiGRU, trained from scratch)
     └── AST + BiGRU Main Model (AudioSet-pretrained AST backbone + BiGRU head)
              ↓
    Timeline Post-Decoder (Tuned strictly on validation data)
              ↓
     AUDIO CONTEXT LAYER (Class, Onset, Offset, Confidence)
              ↓
       QUESTION ROUTER
     ├── Perceptual
     ├── Counting
     ├── Temporal
     ├── Knowledge-Grounded Causal Interpretation
     └── Other (Duration, Overlap, Alert)
              ↓
   Grounded Answer + Timeline Evidence
              ↓
 Evaluation & Edge Feasibility Benchmark
              ↓
       Technical PDF Report
```

- **CRNN Baseline**: Lightweight model trained from scratch; measures feasibility on resource-constrained edge hardware.
- **AST + BiGRU Main Model**: Server-oriented acoustic foundation architecture leveraging frozen AudioSet patch tokens.
- **Oracle Timeline**: Evaluates the reasoner on ground-truth event timelines to diagnose perception vs. reasoning error attribution.

---

## 3. Dataset

The benchmark evaluated in this repository is a **synthetic audio-context benchmark constructed from real ESC-50 recordings** (not plain ESC-50 clips), organized into **700 synthetic scenes: 500 train / 80 validation / 120 test**:

- **Format**: 16 kHz sampling rate, single-channel (mono), exactly 10.0 seconds per scene.
- **Classes**: 22 foreground event classes across 4 stationary acoustic environments (`street`, `home`, `farm`, `celebration`).
- **Splits**:
  - **Train**: 500 scenes | 6,578 QA pairs | ESC-50 Folds 1, 2, 3 (611 unique source clips)
  - **Validation**: 80 scenes | 1,030 QA pairs | ESC-50 Fold 4 (174 unique source clips)
  - **Test**: 120 scenes | 1,558 QA pairs | ESC-50 Fold 5 (190 unique source clips)
  - **Total**: 700 synthetic scenes: 500 train / 80 validation / 120 test | 9,166 total QA pairs: 6,578 train / 1,030 validation / 1,558 test
- **Source-Level Leakage**: **Strictly 0.0%**. No original ESC-50 recording appears across more than one split ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).
- **Boundary Sanitization**: Events maintain a minimum same-class separation of $\ge 0.6$ s and non-ambiguous temporal ordering ($\ge 0.5$ s onset separation for order questions).

---

## 4. Models

### CRNN Baseline (Edge-Oriented)
- **Architecture**: 3-block 2D Convolutional network + 2-layer Bidirectional GRU + Linear classification projection.
- **Inputs**: 64-band Log-mel spectrograms extracted on the fly with SpecAugment masking.
- **Trainable / Total Parameters**: **301,656** (~1.36 MB in FP32).
- **Purpose**: Establishes performance for a self-contained, from-scratch model capable of sub-60 ms CPU execution.

### AST + BiGRU Main Model (Accuracy/Server-Oriented)
- **Architecture**: Audio Spectrogram Transformer (`MIT/ast-finetuned-audioset-10-10-0.4593`) backbone + LayerNorm + Linear + 2-layer BiGRU head.
- **Backbone**: Frozen 86,187,264 parameters; 100 temporal steps (768-dimensional features) cached per scene.
- **Trainable Head Parameters**: **500,502** (~1.95 MB in FP32).
- **Total Parameters**: **86,687,766**.
- **Purpose**: Maximizes acoustic event boundary localization and representation quality using AudioSet pretraining.

---

## 5. Experimental Results

Evaluated on the held-out test split (120 scenes, 1,558 questions). These metrics represent performance on the **medium-scale synthetic ACL benchmark under seed 42**, not unconstrained real-world audio QA.

| Metric | CRNN Baseline | AST + BiGRU Main Model | Question-Only Prior | Uniform Chance | Oracle Timeline (Upper Bound) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Trainable Parameters** | 301,656 | 500,502 | — | — | — |
| **Total Parameters** | 301,656 | 86,687,766 | — | — | — |
| **Test Segment-F1 (1s)** | 0.644 | **0.877** | — | — | 1.000 |
| **Test Event-F1** | 0.460 | **0.724** | — | — | 1.000 |
| **Overall QA Accuracy (Micro)** | **73.9%** | **91.0%** | 40.1% | 34.4% | **100.0%** |
| • Macro Average Over Types | 72.1% | **90.3%** | 38.9% | 33.5% | 100.0% |

- **Question-Only Prior**: 40.1% accuracy (derived strictly from training label distributions; ignores audio).
- **Chance Baseline**: 34.4% across closed-set answer choices.
- **Oracle Baseline**: **100.0%** accuracy across all 1,558 test questions when using ground-truth timelines. The 100% Oracle result demonstrates that, under the benchmark's ground-truth event timeline, the ACL reasoning and QA layer correctly resolves the evaluated questions.

Under the reported configuration and seed (seed 42), the comparison between CRNN and AST+BiGRU shows a substantial difference on the reported test benchmark.

---

## 6. QA Breakdown (AST + BiGRU Main Model)

Performance across individual reasoning categories on the held-out test split:

| Question Category | Test Questions ($n$) | AST + BiGRU Accuracy | Question-Only Prior | Chance Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Perceptual** (Presence, Scene) | 480 | **97.1%** | 37.9% | 33.3% |
| **Counting** | 240 | **83.3%** | 38.3% | 22.1% |
| **Temporal** (First, Order, Before, After) | 404 | **87.9%** | 40.6% | 37.4% |
| **Knowledge-Grounded Causal Interpretation** | 162 | **92.0%** | 23.5% | 27.8% |
| **Other** (Duration, Overlap, Alert) | 272 | **91.2%** | 54.4% | 46.7% |

### Detailed Counting Metrics
- **Exact Match**: **83.3%** (200 / 240 questions)
- **Within-1 Match**: **97.9%** (235 / 240 questions)
- **Mean Absolute Error (MAE)**: **0.188** events

---

## 7. Error Analysis

The 100% Oracle result demonstrates that, under the benchmark's ground-truth event timeline, the ACL reasoning and QA layer correctly resolves the evaluated questions. Consequently, on this benchmark, end-to-end question answering errors are attributable to acoustic perception (detector) inaccuracies rather than reasoning engine failures.

### AST + BiGRU (140 errors out of 1,558 questions = 9.0% error rate)
- **Boundary or Order Errors**: **91** (65.0%) — Detector predicted event presence but slight collar boundary misalignment altered onset ordering or duration comparisons.
- **Missed Events**: **44** (31.4%) — Quiet or overlapping bursts fell below the detection threshold.
- **Spurious False Alarms**: **5** (3.6%) — Background artifacts triggered false event segments.

### CRNN Baseline (406 errors out of 1,558 questions = 26.1% error rate)
- **Missed Events**: **233** (57.4%) — Primary failure mode from training a compact network from scratch on limited scenes.
- **Boundary or Order Errors**: **155** (38.2%)
- **Spurious False Alarms**: **18** (4.4%)

---

## 8. Edge Feasibility & Deployment Profile

Benchmarked on single-thread desktop CPU (`torch.set_num_threads(1)`):

| System Profile | Precision | Storage Size | Latency per 10s Clip | Real-Time Factor (RTF) | Test Event-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CRNN (Edge Baseline)** | FP32 | 1.36 MB | 56.37 ms | 0.0056 (~177× real-time) | 0.460 |
| **CRNN (Dynamic INT8)** | INT8 | **0.74 MB** | **52.34 ms** | **0.0052 (~191× real-time)** | 0.457 (-0.003) |
| **AST + BiGRU (Server)** | FP32 | 1.95 MB (Head) + 346 MB (Backbone) | Server Batch | — | 0.724 |

The CRNN model demonstrates that dynamic INT8 quantization compresses the network below 1 MB with negligible degradation in Event-F1 (-0.003), making it viable for wearable or smartwatch deployment. The AST + BiGRU model serves as a high-accuracy server-side system or teacher model for future distillation.

---

## 9. Reproduction Instructions

### Environment Setup
```bash
# 1. Clone repository
git clone https://github.com/USER/Audio-Context-Layer.git
cd Audio-Context-Layer

# 2. Install dependencies
pip install -r requirements.txt
```

### Run Unit Tests
```bash
python -m pytest -v tests
```
*Expected: 10 passed in ~8 seconds.*

### Reproducing the Pipeline via Modular Scripts

**Windows**:
```cmd
:: Run the complete test suite
scripts\run_all.bat test

:: Evaluate existing runs
scripts\run_all.bat evaluate dataset_mid

:: Measure edge latency and quantization
scripts\run_all.bat edge dataset_mid

:: Compile PDF report
scripts\run_all.bat report dataset_mid "Your Name"
```

**Linux / Colab**:
```bash
bash scripts/run_all.sh test
bash scripts/run_all.sh evaluate dataset_mid
bash scripts/run_all.sh edge dataset_mid
bash scripts/run_all.sh report dataset_mid "Your Name"
```

### Interactive CLI Demo
Run inference on a single audio scene:
```bash
python -m acl.demo --run runs/ast_bigru_medium --wav dataset_mid/audio/test/test_00000.flac
```
*Example Output:*
```text
Scene guess: a celebration or public event
Detected sound events (10 s recording):
- church bells: 1.5s to 5.0s
- clapping: 2.6s to 5.0s
- fireworks: 4.6s to 7.2s
- church bells: 5.5s to 6.0s
- church bells: 7.3s to 8.8s

Automatic questions:
  Q: How many times does the church bells sound occur?  A: 3
  Q: How many times does the clapping sound occur?  A: 1
  Q: Which sound lasts the longest?  A: church bells
```

---

## 10. Dataset Access

Pre-generated synthetic audio scenes, event annotations, and QA records are packaged in the repository under `dataset_mid/` and downloadable at:

[Google Drive Dataset Folder](https://drive.google.com/drive/folders/1X23DqrP6YCT2ScpBJJQj7Jr39Lb_qyXM?usp=drive_link)
(URL: `https://drive.google.com/drive/folders/1X23DqrP6YCT2ScpBJJQj7Jr39Lb_qyXM?usp=drive_link`)

---

## 11. Technical Report

A comprehensive 8-page technical report documenting dataset construction, model loss curves, per-class event F1 breakdowns, and full methodology is available at:

[report/ACL_technical_report_medium.pdf](report/ACL_technical_report_medium.pdf)

---

## 12. Limitations

1. **Synthetic-to-Real Domain Gap**: Scenes are composed of isolated sound bursts placed over stationary acoustic backgrounds without room impulse reverberation, occlusion, or moving sound sources. Performance on real-world unconstrained audio requires further validation.
2. **Closed Ontology**: Limited to the 22 foreground sound classes present in this benchmark.
3. **Template Question Parsing**: The current QA router utilizes regular expression matching aligned with template generators. Open-ended, conversational queries require integration with an LLM parser over the serialized timeline.
4. **Knowledge-Grounded Interpretation**: Causal queries rely on a predefined class-to-cause lookup table and do not model complex, multi-event causal dynamics.
5. **Statistical Scope**: Reported numbers reflect a single experimental run under seed 42. While results show a substantial difference on the reported test benchmark, multi-seed variance has not been evaluated.
6. **No End-to-End Audio LLM Comparison**: Heavy audio-language models (e.g., Qwen2-Audio, Audio Flamingo 2) were not evaluated directly on this synthetic benchmark due to resource constraints.

---

## 13. License & Attribution

- **Source Code**: The software implementation, training pipeline, evaluation tools, and scripts in this repository are licensed under the [MIT License](LICENSE).
- **Dataset & Derived Audio**: The underlying sound recordings originate from the ESC-50 dataset (Karol J. Piczak, 2015), licensed under Creative Commons Attribution-NonCommercial 3.0 (CC BY-NC 3.0). The MIT license applies strictly to the source code and does NOT cover the audio dataset or derived synthetic mixtures. All synthetic audio content is restricted to non-commercial research and educational use under CC BY-NC 3.0.

