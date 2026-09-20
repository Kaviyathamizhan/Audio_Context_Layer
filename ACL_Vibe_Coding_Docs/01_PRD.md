# Document 01 — PRD
# Audio Context Layer (ACL)

## App / System Name
**Audio Context Layer (ACL)**

## Tagline
An end-to-end audio question answering PoC that converts audio into a structured event timeline and answers perceptual, counting, temporal, and knowledge-grounded causal questions.

## Problem
Conventional audio classification identifies sound classes but does not directly provide structured context for questions about what happened, how many events occurred, when events occurred, or how detected events can be interpreted. ACL first creates an Audio Context Layer and then reasons over it.

## Target User
The primary user is a technical evaluator/reviewer assessing an audio ML/AI proof of concept. ACL is a reproducible technical system, not a consumer-facing product.

## Core Value Proposition
ACL separates acoustic event detection from contextual reasoning. Audio is converted into an intermediate event timeline containing class, onset, offset, and confidence, which is then used to answer different question types.

## Must Have
- Use real ESC-50 recordings as source audio.
- Construct synthetic 10-second multi-event scenes.
- Maintain source-level train/validation/test separation.
- Generate ground-truth event timelines.
- Generate perceptual, counting, temporal, and knowledge-grounded causal interpretation QA.
- Preprocess to 16 kHz mono, 10 seconds.
- Main model: AST + BiGRU.
- Baseline: CRNN.
- Tune timeline decoder on validation only.
- Build ACL from onset, offset, class, confidence.
- Route questions to the four supported reasoning types.
- Evaluate SED and QA.
- Include oracle context and train-only question-frequency baseline.
- Use chance only for explicitly closed-set answer spaces.
- Include cluster-bootstrap 95% CIs.
- Perform acoustic error analysis.
- Perform edge feasibility profiling.
- Generate reproducible artifacts and a technical PDF report.

## Nice to Have
No additional product features are defined. Optional work must not change the locked architecture or methodology.

## Out of Scope
- Replacing AST + BiGRU or CRNN with another primary architecture.
- Qwen2-Audio, LLMs, RAG, or unrelated generative models.
- Claims of genuine causal inference.
- Test-set decoder tuning.
- Fabricated or manually entered metrics.
- Treating the synthetic benchmark as a human-annotated real-world QA dataset.
- Database, authentication, or consumer web application.

## User Stories
- As a technical evaluator, I want to provide audio and a natural-language question so ACL can return a grounded answer.
- As a researcher, I want exact synthetic event timelines so SED and reasoning can be evaluated independently.
- As an ML engineer, I want CRNN and AST + BiGRU experiments so acoustic-context performance can be compared empirically.
- As an evaluator, I want oracle context so acoustic errors can be separated from reasoning errors.
- As a reviewer, I want reproducible artifacts so reported metrics can be traced to predictions.

## Success Metrics
Success means the complete benchmark, training, evaluation, error-analysis, edge-profiling, and reporting pipeline executes reproducibly. Final accuracy values are not predetermined and must come only from executed experiments.
