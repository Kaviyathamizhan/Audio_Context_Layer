# Experiment Card: Audio Context Layer (ACL) Medium-Scale Benchmark

---

## 1. Experiment Overview

- **Experiment Identifier**: `ACL-Medium-Seed42`
- **Architecture**: Two-stage Audio Context Layer (SED Perception $\rightarrow$ Structured Event Timeline $\rightarrow$ Question Router $\rightarrow$ Grounded Answer + Evidence)
- **Primary Goal**: Evaluate whether decoupling acoustic sound event detection from deterministic contextual reasoning enables verifiable, explainable audio question answering and exact error attribution.
- **Date Executed**: 2026-09-20
- **Random Seed**: `42` (Fixed for dataset synthesis, train/val/test assignment, and training schedules)

---

## 2. Dataset Specifications

- **Dataset Name**: Synthetic Audio-Context Benchmark Constructed from Real ESC-50 Recordings (`dataset_mid` / `ACL_Dataset`)
- **Source Audio Corpus**: ESC-50 (Karol J. Piczak, 2015, CC BY-NC 3.0)
- **Audio Format**: 16,000 Hz, Single-Channel (Mono), 10.000 seconds, 16-bit FLAC
- **Foreground Classes**: 22 environmental sound event classes
- **Stationary Backgrounds**: 4 acoustic scene contexts (`street`, `home`, `farm`, `celebration`)
- **Split Breakdown**:
  - **Train**: 500 scenes | 6,578 questions | ESC-50 Folds 1, 2, 3 (611 unique clips)
  - **Validation**: 80 scenes | 1,030 questions | ESC-50 Fold 4 (174 unique clips)
  - **Test**: 120 scenes | 1,558 questions | ESC-50 Fold 5 (190 unique clips)
  - **Total**: 700 synthetic scenes: 500 train / 80 validation / 120 test | 9,166 total questions: 6,578 train / 1,030 validation / 1,558 test
- **Source-Level Leakage**: **0.0%** (Zero shared ESC-50 source clips across splits)

---

## 3. Evaluated Models & Checkpoints

### A. CRNN Baseline (From Scratch)
- **Architecture**: 3-block 2D CNN (64-band log-mel) + 2-layer BiGRU (hidden 64) + Linear classifier
- **Trainable / Total Parameters**: 301,656
- **Training Setup**: 25 epochs, OneCycleLR (max lr=0.002), batch size 16, pos_weight=4.0
- **Training Time**: 2,250.0 s (~37.5 min on CPU)
- **Checkpoint Location**: [`runs/crnn_medium/best.pt`](file:///d:/Audio%20ML/runs/crnn_medium/best.pt) (1.35 MB)
- **Optimal Validation Decoder**: `threshold=0.6, min_len=4, merge_gap=2, median=1`

### B. AST + BiGRU Main Model (AudioSet Pretrained)
- **Architecture**: Frozen Audio Spectrogram Transformer (`MIT/ast-finetuned-audioset-10-10-0.4593`) + LayerNorm + Linear (768 $\rightarrow$ 128) + 2-layer BiGRU (hidden 128) + Linear classifier
- **Trainable Head Parameters**: 500,502
- **Frozen Backbone Parameters**: 86,187,264
- **Total Parameters**: 86,687,766
- **Training Setup**: 25 epochs on cached 100-step AST features, OneCycleLR (max lr=0.001), batch size 16, pos_weight=4.0
- **Training Time**: 483.2 s (~8.0 min on CPU with cached representations)
- **Checkpoint Location**: [`runs/ast_bigru_medium/best.pt`](file:///d:/Audio%20ML/runs/ast_bigru_medium/best.pt) (2.01 MB)
- **Optimal Validation Decoder**: `threshold=0.5, min_len=4, merge_gap=2, median=3`

---

## 4. Evaluated Metrics Summary (Held-Out Test Set)

| Metric | CRNN Baseline | AST + BiGRU Main Model | Question-Only Prior | Uniform Chance | Oracle Timeline (Upper Bound) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Segment-F1 (1s)** | 0.644 | **0.877** | — | — | 1.000 |
| **Event-F1** | 0.460 | **0.724** | — | — | 1.000 |
| **Overall QA Accuracy (Micro)** | **73.94%** (1,152 / 1,558) | **91.01%** (1,418 / 1,558) | 40.05% (624 / 1,558) | 34.40% | **100.00%** (1,558 / 1,558) |
| • Perceptual QA | 86.25% (414 / 480) | **97.08%** (466 / 480) | 37.92% | 33.33% | 100.00% |
| • Counting QA | 61.67% (148 / 240) | **83.33%** (200 / 240) | 38.33% | 22.08% | 100.00% |
| • Temporal QA | 67.82% (274 / 404) | **87.87%** (355 / 404) | 40.59% | 37.38% | 100.00% |
| • Causal Interpretation QA | 70.37% (114 / 162) | **91.98%** (149 / 162) | 23.46% | 27.78% | 100.00% |
| • Other (Duration/Overlap/Alert) | 74.26% (202 / 272) | **91.18%** (248 / 272) | 54.41% | 46.69% | 100.00% |
| **Counting Exact Match** | 61.67% | **83.33%** | — | — | 100.00% |
| **Counting Within-1 Match** | 92.08% | **97.92%** | — | — | 100.00% |
| **Counting MAE** | 0.479 | **0.188** | — | — | 0.000 |

Under the reported configuration and seed (seed 42), the comparison between CRNN and AST+BiGRU shows a substantial difference on the reported test benchmark.

---

## 5. Edge Feasibility Profile

- **CRNN FP32**: Size = 1.36 MB, Latency = 56.37 ms per 10s audio, Real-Time Factor (RTF) = 0.0056 (177× faster than real-time), Event-F1 = 0.460.
- **CRNN Dynamic INT8**: Size = **0.74 MB** (fits in <1 MB memory limit), Latency = **52.34 ms**, RTF = **0.0052** (191× faster than real-time), Event-F1 = **0.457** (-0.003 difference).
- **AST Server Trade-off**: Trainable head = 1.95 MB, Frozen backbone = ~346 MB.

---

## 6. Error Analysis & Attribution

- **Perception vs. Reasoning Attribution**: The 100% Oracle result demonstrates that, under the benchmark's ground-truth event timeline, the ACL reasoning and QA layer correctly resolves the evaluated questions. On this benchmark, end-to-end question answering errors are attributable to acoustic perception (detector) inaccuracies rather than reasoning engine failures.
- **AST + BiGRU Error Breakdown (140 wrong / 1,558 questions)**:
  - Boundary or Order Misalignment: **91** (65.0%)
  - Missed Sound Events: **44** (31.4%)
  - Spurious False Detections: **5** (3.6%)
- **CRNN Error Breakdown (406 wrong / 1,558 questions)**:
  - Missed Sound Events: **233** (57.4%)
  - Boundary or Order Misalignment: **155** (38.2%)
  - Spurious False Detections: **18** (4.4%)

---

## 7. Limitations & Scope

1. **Synthetic Audio**: Evaluated on multi-event synthetic compositions derived from real ESC-50 bursts. Performance on unconstrained, reverberant, or overlapping continuous speech/audio requires further domain transfer study.
2. **Ontology Coverage**: 22 environmental classes.
3. **Causal Interpretation**: Uses a structured ontology mapping table; does not claim mathematical counterfactual causal discovery.
4. **Statistical Scope**: Reported numbers reflect a single experimental run under seed 42. While results show a substantial difference on the reported test benchmark, multi-seed variance has not been evaluated.

---

## 8. License & Terms

- **Code**: MIT License (applies strictly to source code and scripts).
- **Audio Dataset**: Built upon ESC-50 sound recordings (Karol J. Piczak, 2015), governed separately by Creative Commons Attribution-NonCommercial 3.0 (CC BY-NC 3.0). The MIT license does NOT apply to the audio dataset.
