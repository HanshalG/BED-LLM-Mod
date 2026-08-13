# SymptomCheck dynamic-support source result

Date: 2026-08-13

Status: **source failed closed; zero model calls**

## Result

The pinned SymptomCheck Avey release passes immutable provenance, exact
400-case population, exact eight-field schema, row uniqueness, diagnosis
diversity, visible/private separation, released patient behavior, simulator
behavior, and deterministic partition gates.

It fails the conjunctive requirement that every required field be structurally
nonempty. Aggregate localization shows that seven fields are nonempty in all
400 cases, while `social_history` is empty in all 400. Therefore no row passes
the frozen all-fields-nonempty rule.

This is another source-schema null rather than evidence against the released
public/private patient interface. Treating social history as optional after
observing the audit, deleting the field, or selecting a repaired projection
would weaken the frozen gate. The exact SymptomCheck construction is closed. A
new upstream release or a genuinely new prospective task definition would be
required for reconsideration.

## Privacy and accounting

No individual case ID, demographics, complaint, history, finding, diagnosis,
dialogue, or endpoint was serialized. Mechanics, opportunity, development, and
confirmation were not opened.

- OpenRouter calls: `0`
- OpenRouter cost: `$0`
- OATML cluster use: none

## Bindings

- protocol:
  `results/nonmyopic/SYMPTOMCHECK_DYNAMIC_SUPPORT_SOURCE_PROTOCOL_20260813.md`
- manifest:
  `results/nonmyopic/symptomcheck_dynamic_support_source/MANIFEST.json`
- audit:
  `results/nonmyopic/symptomcheck_dynamic_support_source/SOURCE_AUDIT.json`
- implementation: `scripts/symptomcheck_dynamic_support_source_audit.py`
- tests: `tests/test_symptomcheck_dynamic_support_source_audit.py`

This result authorizes nothing.
