# Document 06 — Implementation Plan
# Audio Context Layer (ACL)

## Phase 1 — Repository Foundation
Maintain `acl/`, tests, scripts, notebooks, configs, dataset, runs, and report structure. Verify imports, CLI modules, configurations, and tests.

## Phase 2 — Dataset Integrity
Use ESC-50 folds 1–3 for train, 4 for validation, and 5 for test. Generate synthetic 10-second scenes, ground-truth timelines, and QA. Verify source-level leakage and metadata consistency.

Medium target: 500 train / 80 validation / 120 test scenes.
Full target: 1,600 train / 300 validation / 500 test scenes.
Use actual generated counts in reports.

## Phase 3 — Smoke Verification
Run the tiny generator, short CRNN training, prediction, validation-only decoding, evaluation, edge profiling, report generation, and CLI demo.

## Phase 4 — Medium CRNN
Train the CRNN baseline on 500/80/120. Save checkpoint, configuration, predictions, metrics, plots, and results. Evaluate SED, QA, and errors.

## Phase 5 — Medium AST + BiGRU
Use the existing AST + BiGRU implementation. Keep AST frozen according to the locked implementation and cache features where supported. Save complete run artifacts.

## Phase 6 — Decoder Tuning
Search decoder parameters on validation only. Freeze the selected configuration before test evaluation.

## Phase 7 — Evaluation
Compute Segment-F1, Event-F1, per-class metrics, QA accuracy by type, overall accuracy, counting exact/within-1/MAE, oracle condition, train-only question-frequency baseline, applicable chance baseline, and cluster-bootstrap 95% CIs.

## Phase 8 — Error Analysis
Use actual predictions to classify missed_event, spurious_event, and boundary_or_order. Inspect QA failures and trace them to acoustic detection, localization, decoding, reasoning, or knowledge-grounded interpretation.

## Phase 9 — Edge Feasibility
Measure parameters, FP32 size, INT8 size where supported, CPU single-thread latency, and real-time factor. Keep CRNN and AST + BiGRU results distinct.

## Phase 10 — Full-Scale Experiment
After medium validation, use Colab/Kaggle GPU for 1,600/300/500. Generate and verify data, cache AST features where supported, train both models, evaluate, analyze errors, profile edge feasibility, and save final artifacts.

## Phase 11 — Technical Report
Generate the PDF from actual artifacts. Include problem, dataset construction, timeline, QA, architecture, models, decoder, evaluation, baselines/oracle, results, errors, edge feasibility, and limitations.

Use the terms:
- “synthetic audio-context benchmark constructed from real ESC-50 recordings”
- “knowledge-grounded causal interpretation”

Do not claim genuine causal inference.

## Phase 12 — Final CLI Demonstration
Run:
```text
Audio
 ↓
Preprocessing
 ↓
AST + BiGRU event detection
 ↓
Timeline decoder
 ↓
Audio Context Layer
 ↓
Question router
 ↓
Reasoning
 ↓
Answer + evidence
```

## Global Done Criteria
- Dataset reproducible.
- Source leakage zero.
- Timelines and QA valid.
- CRNN trained/evaluated.
- AST + BiGRU trained/evaluated.
- Decoder tuned only on validation.
- Oracle and train-only prior evaluated.
- Applicable chance baseline evaluated.
- SED/QA/counting metrics generated.
- Bootstrap CIs generated.
- Error analysis generated from predictions.
- Edge profiling completed.
- Results stored as artifacts.
- Final report generated from artifacts.
- CLI demo works.

## Non-Negotiable
Implemented ≠ Executed ≠ Validated.
Never fabricate metrics, manually alter results, or change the locked architecture to improve an isolated result.
