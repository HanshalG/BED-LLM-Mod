# Bongard History-Blind Estimand Audit

Date: 2026-08-08

Status: prospective claim correction; zero model calls and zero endpoint reads.

## Finding

The unopened `dynamic_depth2` versus `history_blind_depth2` control changes
the simulated belief support used to score first queries. It does not change
the updater deployed after the realized first answer:

- dynamic planning uses answer-conditioned support regeneration for each
  simulated first answer;
- history-blind planning uses a same-seed fresh support generated without that
  simulated answer, then applies the same analytical answer update;
- after the real first answer, both arms use the same answer-conditioned branch
  to choose query two and the same final-history regeneration.

The endpoint contrast therefore identifies the value of answer-conditioned
simulation for first-query planning under a common realized updater. It does
not identify a causal effect of deploying an answer-conditioned rather than a
history-blind belief updater.

## Prospective Repair

The clarification
`BONGARD_OPENWORLD_HISTORY_BLIND_ESTIMAND_CLARIFICATION_20260808.md`
narrows all allowed claim text to the executable estimand. It changes no task,
image, prompt, request, seed, policy action, endpoint, gate, threshold, call
cap, or cost cap. Regression tests now assert that both arms share the realized
answer-conditioned branch and reject the broader updater claim in generated
reports and paper fragments.

The rebound frozen chain is:

| Artifact | SHA-256 |
|---|---|
| Estimand clarification | `65a6e901dc815d1603611e180b0abdf728e07d1a452e0c442f48fbb1812aea10` |
| Development manifest V15 | `2c0be4cc4aaaa66bab715386ceeb9bb9fb2e9c06545ee283be9b9e7a47d27835` |
| Confirmation manifest V12 | `1c52dac31b82281d1ad057729bb462353f18cf323900c87d7747419681246078` |
| Naive first-link manifest V6 | `65d4b497fee8f00cd9e1794054dc191dfaced2d5aa1fd172fb0717a40023eca9` |
| Claim generator v5 | `923bcbf9c347a73708ce9fdedfb60926b78a776ca15a15fa597150121e49e85e` |
| Paper-fragment protocol | `fb54ceb1c18a5b6f6b00ec2eb76ab8325f9fbb4cae059b5d7b56e372d6a59342` |
| Paper-fragment generator | `389fb8670ec686c463fbea3c7a1036a3a51a1f4e3c169d400195c9a364b6516b` |

The banked naive smoke remains valid through its exact V1-to-V2 replay
certificate (`725bb7827508b6b20bf34037cc33feb17b78e1c31c4da9a21d9c3b9ea474d1da`).
Its replay context deliberately retains the predecessor Development V14 hash;
the new Naive V6 manifest binds that certificate into the current V15 chain.

## Verification

- Development V15 verifier: pass.
- Confirmation V12 independent verifier: pass.
- Naive V6 verifier and banked smoke replay: pass.
- Deterministic paper binding verifier: pass for all 11 bound files.
- Full Bongard regression suite: `165 passed in 238.88s`.
- Authenticated August 10 preflight: `ready_without_paid_calls`, with zero model
  calls, zero files written, and all execution paths absent.
- Live OpenRouter balance during preflight: `$24.878986213`; the reported new
  `$30` was not yet posted by the credits endpoint.

No scientific result was viewed or generated during this audit. The current
manuscript remains byte-identical at
`6ece61e00c284c961e08375b873a410a959bf9bab978f847c0d099dbaf7453bf`.
