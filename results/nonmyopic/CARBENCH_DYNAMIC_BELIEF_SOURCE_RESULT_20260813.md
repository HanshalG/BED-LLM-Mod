# CAR-bench dynamic-belief source result

Date: 2026-08-13

Status: **source failed closed; zero model calls**

## Result

The frozen CAR-bench source audit passes every provenance, population, schema,
identity, task-type, released-code, and split gate. It fails the conjunctive
`required_structural_fields_nonempty` gate.

The failure is caused by the protocol's recursive interpretation of nonempty:
all 31 official train rows and all 25 official test rows contain nonempty
personas, instructions, disambiguation notes, and task-appropriate ambiguity
elements, but every serialized `context_init_config` and `actions` structure
contains at least one optional empty leaf. Under the implementation frozen for
this audit, one such leaf makes the complete structure fail.

This is an audit-design null, not evidence that CAR-bench lacks informative
disambiguation tasks. The exact construction is nevertheless closed. Weakening
the recursive rule after observing the failure, selecting a subset, or opening
mechanics under a repaired interpretation would violate the frozen boundary.
A future CAR-bench route requires a genuinely new prospective protocol or a new
upstream release; it cannot be called a rerun of this source audit.

## Privacy and accounting

No individual task ID, persona, instruction, context value, action,
disambiguation element, user answer, or endpoint was serialized. Mechanics,
opportunity, development, and confirmation were not opened.

- OpenRouter calls: `0`
- OpenRouter cost: `$0`
- OATML cluster use: none

## Bindings

- protocol:
  `results/nonmyopic/CARBENCH_DYNAMIC_BELIEF_SOURCE_PROTOCOL_20260813.md`
- manifest:
  `results/nonmyopic/carbench_dynamic_belief_source/MANIFEST.json`
- source audit:
  `results/nonmyopic/carbench_dynamic_belief_source/SOURCE_AUDIT.json`
- implementation: `scripts/carbench_dynamic_belief_source_audit.py`
- tests: `tests/test_carbench_dynamic_belief_source_audit.py`

This result authorizes nothing.
