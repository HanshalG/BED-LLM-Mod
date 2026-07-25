# Ambig-IaC V2 Particle-Schema Serving Smoke Result

Date: 2026-07-25

## Outcome

**The final V2 serving gate failed. Ambig-IaC is closed for the current project,
with no V3 and no fresh-task efficacy run.**

All three calls completed cleanly, but only task 272 passed the frozen exact
parser. Tasks 66 and 156 represented `depends_on` as a list rather than the
required scalar label. Treating either form as equivalent after observing the
responses would be a posthoc parser amendment forbidden by the preregistration.

No target plan, branch, scientific score, answer, or endpoint was accessed.

## Frozen Execution

| Metric | Result |
|---|---:|
| Physical requests | 3 |
| HTTP attempts | 3 |
| Transport retries | 0 |
| Reasoning tokens | 0 |
| Forced exits | 0 |
| Exact valid responses | 1/3 |
| Task 272 | valid, 2 resources, 10 features, 3 dimensions |
| Task 66 | invalid scalar `depends_on` contract |
| Task 156 | invalid scalar `depends_on` contract |
| Target plans opened | 0 |
| Scientific endpoints | 0 |
| Cost | `$0.00412425` |

Every response had exactly the three V2 top-level keys. All resource rows used
string labels and addresses, and all attribute rows used string labels plus key
lists. The two invalid responses used string dependency sources but dependency
lists in every dependency row. This is a narrow schema mismatch, not an
efficacy measurement.

## Decision

The preregistration explicitly named this the final schema-only gate and ruled
out a V3. The exact V2 interface therefore closes. The next experiment must use
a different environment or an independently justified interface, not reissue
these prompts with a list-tolerant parser.

All 123 eligible Ambig-IaC tasks outside the three V1/V2 prompts remain
scientifically untouched: their model particles, questions, scores, and
endpoints were never generated or viewed.

## Integrity And Budget

- Preregistered commit: `52aa08c`.
- Run ID: `ambig-iac-schema-v2-smoke-20260725T084346Z`.
- Model: `openai/gpt-5.4-mini`, non-thinking, temperature `.7`.
- Private raw SHA-256:
  `d98ff60057911281815d657a36d8556073865dbcad180385d8092ed2052948bb`.
- Public artifact:
  `results/nonmyopic/ambig_iac_schema_v2_smoke/ambig-iac-schema-v2-smoke-20260725T084346Z/SMOKE.json`.
- Project-ledger spend after V2: `$86.18245031920742`.
- Monday new-work allowance remaining: `$14.9608545`.
- Authenticated OpenRouter endpoint immediately after the run still reported
  `$44.206476634` remaining; provider credit accounting had not yet reflected
  the `$0.00412425` charge.
- OatML resources used: none.
