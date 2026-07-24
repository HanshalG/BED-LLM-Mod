# PAPRIKA Murder-Belief Opportunity Result

Date: 2026-07-25
Seed: `24329`
Status: **opportunity conjunction failed; no planner or reserve evaluation**.

## Execution

The two-case serving smoke passed exactly 126 calls with zero reasoning tokens,
six distinct first responses on each case, valid eight-suspect beliefs, and
replay gaps `.02/.01`. It cost `$0.35107525`.

The fresh six-case opportunity screen completed exactly 378 calls:
192 GPT-5.4 belief/action calls and 186 GPT-5.4 Mini environment/equivalence
calls. There were zero reasoning tokens, retries, forced exits, parser failures,
or runtime failures. It cost `$1.09859400`.

## Frozen gates

| Gate | Threshold | Result | Pass |
|---|---:|---:|:---:|
| Mean initial truth probability | at most `.35` | `.1517` | yes |
| Mean distinct first responses | at least `4.0` | `6.0` | yes |
| One-step spread at least `.10` | at least 4/6 | 2/6 | **no** |
| Oracle first differs from greedy | at least 2/6 | 2/6 | yes |
| Pair gain at least `.10` | at least 3/6 | 5/6 | yes |
| Mean pair gain | at least `.10` | `.1283` | yes |
| Non-myopic gap at least `.10` | at least 2/6 | 0/6 | **no** |
| Mean non-myopic gap | at least `.07` | `.0133` | **no** |
| Mean / max replay gap | at most `.10` / `.25` | `.0183` / `.0500` | yes |

All execution and remaining scientific gates passed.

## Mechanism

The generated investigations were concrete and coherent: decrypting messages,
checking access logs, interviewing witnesses separately, testing poison traces,
and following specific contradictions. Every case produced six distinct
first-turn environment responses. Two-step evidence improved truth probability
over the best one-step belief by `.1283` on average.

The improvement was not specifically non-myopic. After nearly every first
response, the unrestricted semantic follow-up generator could pivot to a
decisive records check, interview, or forensic test using the full public scene.
The best continuation after the realized-greedy first action therefore nearly
matched the global two-step oracle. The two cases with different oracle roots
had gaps of only `.03` and `.05`; the other four had zero.

Soft instructions to use newly revealed information do not create a real
prerequisite. A strong LLM can repair an initially weaker investigation on turn
two whenever all actions remain available.

## Decision

Close this exact PAPRIKA murder interface. Do not run reserve cases, adjust
thresholds, restrict follow-ups after seeing outcomes, or repair prompts.

The next viable environment must enforce affordances outside the language model:
tool identifiers, permissions, movement, delayed tests, or irreversible state.
The semantic hypothesis space can remain LLM-generated, but the environment
itself must make some continuations unavailable before an enabling observation.

Artifacts:

- `results/nonmyopic/paprika_murder_belief_smoke/paprika-murder-belief-smoke-20260725T001000Z/SERVING_SMOKE.json`
- `results/nonmyopic/paprika_murder_belief_opportunity/paprika-murder-belief-opportunity-20260725T001500Z/OPPORTUNITY.json`
