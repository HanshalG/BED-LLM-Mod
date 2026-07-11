# Paprika Step 1 Endpoint Audit

> **INVALID-ENDPOINT RECORD:** This audit documents why all neighboring Step 1 policy results are quarantined and must not be cited as policy evidence.

## Verdict

The completed Paprika Step 1 endpoint is not faithful enough to the released
customer-service success protocol to support a claim about EIG versus naive.
The measured difference is dominated by inconsistent success labeling, not
only policy quality. This strengthens the pre-registered STOP-and-discuss
decision after the rescue, rather than authorizing a new run.

## Concrete Evidence

| Task | Arm | Agent action / customer reply | Recorded outcome | Why it matters |
|---|---|---|---|---|
| `0005` trailer connector | naive | The customer says the plug was loose, they tightened it, and the lights work. | Not solved; the identical next query gets `Goal reached`. | Same trajectory changes from unsolved to solved without new information or a new action. |
| `0005` trailer connector | Full2 | Customer says they made the connector secure and that fixed it. | Not solved. | An explicit success report was missed. |
| `0009` kiosk screen | canonical EIG | `Clean the screen surface with a microfiber cloth.` | Customer says cleaning was already tried and failed; not solved. | The action is the released private remedy, but the simulated reply contradicts it. |
| `0009` kiosk screen | generation-thinking rescue | `Clean the display using a microfiber cloth and an approved non-abrasive cleaning solution.` Customer says only `I'll try that right now.` | Solved. | A prospective, unconfirmed action became a false positive. |

The canonical EIG and rescue differ on task `0009` despite essentially the
same remedy. This alone can change a 3-versus-4 resolution comparison by a
full task, which is larger than the observed paired effect.

## Protocol Mismatch

Paprika's released runner calls its customer-service judge on the complete
agent conversation every turn and marks success when either the environment
returns `Goal reached` or that full-conversation judge accepts the trajectory.
See [game.py](/Users/hanshalgoyal/BED-LLM-Mod/external/paprika/llm_exploration/game/game.py:690)
and [game.py](/Users/hanshalgoyal/BED-LLM-Mod/external/paprika/llm_exploration/game/game.py:704).

The adapter instead:

- accepts a literal `Goal reached` reply immediately;
- invokes a separate judge only for an action classified as `solution`;
- passes only the latest query and reply to that judge; and
- skips the judge when the reply appears to report a failed attempt.

See [env.py](/Users/hanshalgoyal/BED-LLM-Mod/environments/paprika_customer_service/env.py:617)
and [prompts.py](/Users/hanshalgoyal/BED-LLM-Mod/environments/paprika_customer_service/prompts.py:81).
That is a material endpoint divergence. It creates false negatives for
diagnostic turns that elicit an explicit fix and false positives when a
semantically correct corrective action is merely promised.

## Consequence

The Step 1 result should be treated as an endpoint-integration failure plus
an insufficient policy comparison, not evidence that EIG loses on a faithful
Paprika benchmark. The original Step 0 requirement was a ground-truth success
check usable for evaluation; this audit shows that requirement has not been
demonstrated for the implemented adapter.

## Only Authorized Next Decisions

1. Stop Path E and retain this audit with the null/insufficient result.
2. Explicitly authorize a new path that first repairs the endpoint to match
   Paprika's complete-conversation success protocol, re-runs Step 0 acceptance
   checks, freezes a new analysis plan, and only then re-runs a fresh paired
   Step 1. This would be a new pre-registered experiment, not a continuation
   of the already spent rescue allowance.
