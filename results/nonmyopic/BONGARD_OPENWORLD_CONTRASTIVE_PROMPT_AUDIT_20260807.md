# Bongard Contrastive-Prompt Audit

Date: 2026-08-07. Model calls: 0. Cost: $0. Scientific endpoints opened: 0.

## Decision

Before any Bongard model response, clarify the semantic-belief prompt to state
the official class relation: a valid visual rule is present in positive
examples and absent from negative examples, and both classes must be used
contrastively.

This is a task-validity clarification, not an outcome-conditioned prompt
change. It reveals no hidden concept, label, image role, source path, candidate,
or endpoint. Tasks, images, observed histories, policies, controls, seeds,
request counts, budgets, dates, endpoints, thresholds, and statistical gates
are unchanged. The mechanics gate remains sensitive at one changed root; it is
not tightened before data.

## Frozen Bindings

- Amendment: `BONGARD_OPENWORLD_LUNA_CONTRASTIVE_PROMPT_AMENDMENT.md`, SHA-256
  `9073815f7968a0fa5e52408012bf75d9fe0b34c8428572620a4975a155bfd82b`.
- Development manifest: superseded
  `8659fb5fc6a02ddc59eb7147b6663d1fef3f96880e29f5e6de9bc0386f8e24aa`;
  authoritative
  `451177a86b8ffbff128c4d8f94d7e6903873ce43050521119721f43882ecc9a4`.
- Confirmation manifest V2: superseded
  `1613bd4f1978a0346ca4cc7fe7511ef99eab26fe8963465b2823fc1d6120f5e1`;
  authoritative
  `34f2c992a1bb8f56d3f882b804caf1fa95c1017e22f9673a7584a0c2a610eee9`.
- The superseded confirmation V1 manifest remains an immutable historical
  artifact and is not authoritative for execution.

## Verification

- Focused prompt test: 8 passed.
- Focused execution and confirmation suite: 76 passed.
- Full Bongard suite: 113 passed in 42.46 seconds.
- Independent confirmation verifier: all 12 checks pass.
- Confirmation execution verifier: exact protocol/core/daily/amendment chain
  passes.
- Authenticated August 10 preflight: `ready_without_paid_calls`; Luna supports
  image/text structured output at live `$0.10/$0.60` per million input/output
  tokens; balance `$24.886393846`; component-cap sum `$2.00`; model calls 0;
  files written 0.

The next permitted action remains the frozen August 10 Luna serving and
mechanics sequence. No model substitution or early development access is
authorized.
