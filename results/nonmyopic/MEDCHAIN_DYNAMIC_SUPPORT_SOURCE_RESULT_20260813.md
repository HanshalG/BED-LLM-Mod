# MedChain dynamic-support source result

Date: 2026-08-13

Status: **source failed closed; zero model calls**

## Result

MedChain passes the immutable code/data binding and every released-interface
contract checked by the frozen audit. Its doctor receives only the chief
complaint, while its patient is grounded in private history and examination
fields and is instructed to withhold unasked examinations and never invent
missing facts. This is the right information architecture for the project.

The exact source nevertheless fails two preregistered population gates. The
pinned data object contains 2,362 cases rather than the advertised and frozen
12,163. Of those, 1,280 (54.19%) satisfy the complete patient-interface shape,
below the frozen 90% fraction. The eligible cases are otherwise substantial and
diverse: they contain 745 distinct normalized diagnosis labels and support the
complete deterministic split.

Because population size and eligibility fraction were conjunctive gates, this
exact construction is closed. Selecting the 1,280 eligible cases after seeing
the mismatch, lowering the fraction, or substituting an external MedChain
dataset would be a post-audit repair. A new immutable upstream release may be
screened only under a new prospective protocol.

This is a source-release null, not evidence against the MedChain patient
interface or against non-myopic clinical questioning.

## Privacy and accounting

No case key, individual case ID, complaint, history, examination, department,
diagnosis, treatment, image, or endpoint was serialized. No mechanics case was
opened.

- OpenRouter calls: `0`
- OpenRouter cost: `$0`
- OATML cluster use: none

## Bindings

- protocol:
  `results/nonmyopic/MEDCHAIN_DYNAMIC_SUPPORT_SOURCE_PROTOCOL_20260813.md`
- manifest:
  `results/nonmyopic/medchain_dynamic_support_source/MANIFEST.json`
- audit:
  `results/nonmyopic/medchain_dynamic_support_source/SOURCE_AUDIT.json`
- implementation: `scripts/medchain_dynamic_support_source_audit.py`
- tests: `tests/test_medchain_dynamic_support_source_audit.py`

This result authorizes nothing.
