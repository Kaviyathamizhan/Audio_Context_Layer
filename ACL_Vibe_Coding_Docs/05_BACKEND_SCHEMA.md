# Document 05 — Backend Schema
# Audio Context Layer (ACL)

## Scope
ACL does not define a production backend, relational database, authentication system, or user-account model. Persistent state is represented by dataset files and ML experiment artifacts.

Therefore database tables, auth, roles, RLS, and production API schemas are **not applicable** to the locked architecture.

## Data Model
### Source Dataset
ESC-50 source recordings and metadata.

### Synthetic Scene
- scene ID
- audio artifact
- source recording references
- scene/event metadata
- ground-truth timeline

### Ground-Truth Event
- event class
- onset
- offset

### Question
- question ID
- scene reference
- question
- question type
- generated ground-truth answer
- answer metadata where implemented

Supported types:
- perceptual
- counting
- temporal
- knowledge-grounded causal interpretation

### Model Run
- configuration
- checkpoint
- validation predictions
- test predictions
- metrics
- plots
- result artifacts

## Artifact Structure
```text
dataset/
├── train/
├── val/
└── test/

runs/<run_name>/
├── config.json
├── checkpoints/best.pt
├── predictions/
├── metrics/
├── plots/
└── results/

report/
└── <report>.pdf
```

## Relationships
```text
ESC-50 source clip
        ↓
Synthetic scene
        ↓
Ground-truth event timeline
        ↓
QA records

Synthetic scene
        ↓
Model prediction
        ↓
Predicted event timeline
        ↓
QA answer
```

## Integrity
- No source clip leakage across splits.
- Unique scene IDs.
- QA scene IDs resolve.
- onset < offset.
- events remain within 10 seconds.
- class ordering is consistent.
- prediction dimensions and IDs match evaluation contracts.

## Auth / Roles / RLS / Webhooks / Production API
Not applicable.

Do not introduce a database or authentication layer as part of this PoC.
