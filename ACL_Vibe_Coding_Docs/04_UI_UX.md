# Document 04 — UI/UX Design Brief
# Audio Context Layer (ACL)

## Scope
The locked ACL architecture defines a CLI inference demo, experiment outputs, figures, and a technical PDF—not a conventional web/mobile frontend.

## Aesthetic
Technical, minimal, research-oriented, and readable.

Priorities:
- event timelines
- model/evaluation metrics
- experiment provenance
- clear predicted-vs-oracle distinction
- readable technical figures

## Primary Interaction
Input:
```text
Audio clip + natural-language question
```

Output:
- detected event timeline
- answer
- supporting context/evidence where implemented

## Event Timeline
Display:
```text
class | onset | offset | confidence
```

## Evaluation Outputs
- `qa_accuracy.png`
- `sed_per_class.png`
- `results.json`
- error examples
- human-readable evaluation summary

## Technical Report
Include architecture, dataset methodology, models, evaluation, results, error analysis, edge feasibility, and limitations.

## Visual Constraints
No product-specific color palette, typography system, or web design system is defined by the locked architecture. Do not invent one as a system requirement.

No responsive web UI is defined.

Textual outputs should remain readable, expose timestamps explicitly, and not rely on color alone.
