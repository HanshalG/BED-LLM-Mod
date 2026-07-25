# Orchid Executable-Particle Opportunity Result

Date: 2026-07-25

## Outcome

**The opportunity gate failed. No path-dependent branch generation is
authorized, and the exact Orchid route is closed.**

All 16 GPT-5.4 Mini responses produced valid executable programs, but the
particle behavior collapsed: every candidate clarification input had exactly
one output across all eight particles on both tasks. Immediate EIG was therefore
zero everywhere. There was no root-ranking problem for non-myopia to solve.

## Frozen Results

| Metric | Task 97 | Task 133 | Gate |
|---|---:|---:|---:|
| Valid particles | 8/8 | 8/8 | at least 6 each |
| Initial held-out pass fraction | `.25` | `1.00` | `.10`-`.90` each |
| Informative queries, EIG >= `.30` | 0/4 | 0/6 | at least 2 each |
| Query endpoint range | `.25` | `1.00` | at least `.10` each |
| Myopic endpoint | `.25` | `1.00` | descriptive |
| Oracle endpoint | `.25` | `1.00` | descriptive |
| Oracle gain over initial | `.00` | `.00` | at least `.10` each |
| Oracle gap over myopic | `.00` | `.00` | mean at least `.05` |
| Myopic root equals oracle root | yes | yes | differ at least once |

Mean initial, myopic, and oracle endpoints were `.625`, `.625`, and `.625`.
The initial-unsaturation, informative-query, oracle-gain, root-difference, and
oracle-gap gates failed.

The positive endpoint ranges do not indicate useful uncertainty. Some exact
target query outputs were absent from the unanimous particle support, yielding
an empty posterior and endpoint zero, while all supported target outputs
retained all eight particles. Because every output partition was degenerate,
EIG could not distinguish these cases prospectively.

## Interpretation

Orchid is intentionally ambiguous at the wording level, but on the frozen
Vagueness prompts GPT-5.4 Mini generated behaviorally identical
implementations despite independent sampling and explicit interpretation
indices. One task's consensus implementation was mostly wrong and the other's
was fully correct; neither represented an uncertain executable belief.

This cleanly reproduces the support-collapse problem seen in prior code routes:
natural-language ambiguity alone does not guarantee diverse model-induced
behavior. Increasing depth over a collapsed support would add calls without
creating a non-myopic decision problem.

Per preregistration, there is no alternate ambiguity-type search, task subset,
temperature rerun, or model rerun on this exact route.

## Integrity And Cost

- Preregistered commit: `bf12b0d`.
- Run ID: `orchid-code-particle-opportunity-20260725T085555Z`.
- Orchid commit: `55ddeb0d3670d22d420e072816cf4f034cf62caf`.
- Model: `openai/gpt-5.4-mini`, non-thinking, temperature `.7`.
- Requests/attempts: 16/16.
- Retries/reasoning tokens/forced exits: 0/0/0.
- Branch-regeneration calls: 0.
- Cost: `$0.00687`.
- Private raw SHA-256:
  `2c3cb4aa377e9b56da05e582db415dc38b20ca45288023adcc20c4c078a0d529`.
- Public artifact:
  `results/nonmyopic/orchid_code_particle_opportunity/orchid-code-particle-opportunity-20260725T085555Z/OPPORTUNITY.json`.
- Project-ledger spend after the run: `$86.18932031920745`.
- Monday new-work allowance remaining: `$14.9539845`.
- Authenticated OpenRouter remaining: `$44.202352384`, or `$19.202352384`
  above the protected `$25` reserve.
- OatML resources used: none.
