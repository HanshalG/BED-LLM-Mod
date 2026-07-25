# Ambig-IaC Regenerated-Support First-Link Smoke Result

Date: 2026-07-25

## Outcome

**The smoke failed closed during initial particle parsing. No branch generation,
score, target-plan load, or scientific endpoint occurred.**

The exact interface is closed. The failed responses will not be repaired,
reparsed under a relaxed schema, or reissued. A materially distinct serving
interface must first pass a separately frozen schema-only test before any fresh
Ambig-IaC tasks are used for efficacy.

## Frozen Execution

| Metric | Result |
|---|---:|
| Physical requests | 15 |
| HTTP attempts | 15 |
| Transport retries | 0 |
| Reasoning tokens | 0 |
| Forced exits | 0 |
| Initial valid particles, task 272 | 2/5 |
| Initial valid particles, task 66 | 1/5 |
| Initial valid particles, task 156 | 5/5 |
| Branch requests | 0 |
| Target plans loaded | 0 |
| Scientific scores/endpoints | 0 |
| Cost | `$0.03502125` |

The frozen minimum was four valid particles in every population. Task 272
therefore triggered the fail-closed stop before the branch batch was built.

## Private Parse Diagnosis

The public artifact retains only aggregate validation errors. Private raw
inspection found:

- task 272: three responses encoded each resource address as a nested object
  instead of the required string;
- task 66: three responses used a different nested resource-object shape and
  one response was malformed JSON; and
- task 156: all five responses followed the frozen schema.

The likely cause is that GPT-5.4 Mini interpreted the prompt's illustrative
`required_output.resources` object as a schema template and reproduced nested
objects. Those completions are semantically plausible Terraform
interpretations, but they are invalid under the preregistered exact parser.
Accepting or normalizing them after inspection would be an unregistered repair.

This is a serving/schema null, not evidence for or against regenerated-support
depth-two ranking. All efficacy gates are unmeasured.

## Integrity And Budget

- Preregistered commit: `d0ef0bd`.
- Run ID: `ambig-iac-first-link-smoke-20260725T083649Z`.
- Model: `openai/gpt-5.4-mini`, non-thinking.
- Source commit:
  `4b50c142ed4638a4caaee9ca0b92d8d0e5b8c8cb`.
- Private raw SHA-256:
  `5bec12283276bfbae2bb507d6af47de07328e6d1c2946d624aaa2383f387ba76`.
- Public failure artifact:
  `results/nonmyopic/ambig_iac_first_link_smoke/ambig-iac-first-link-smoke-20260725T083649Z/SMOKE_FAILURE.json`.
- Project-ledger spend after the run: `$86.17832606920743`.
- Local `$15` allowance remaining: `$14.96497875`.
- Live OpenRouter remaining after the run: `$44.206476634`.
- Live amount above the protected `$25` reserve: `$19.206476634`.
- OatML resources used: none.
