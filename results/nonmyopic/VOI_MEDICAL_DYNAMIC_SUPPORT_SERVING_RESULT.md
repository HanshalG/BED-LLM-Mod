# VoI Medical Dynamic-Support Serving Result

## Decision

All preregistered serving and relevance gates passed. The frozen initial
supports and semantic roots may be reused without reissue in a separately
preregistered path-dependent tree mechanics test on the same five mechanics
rows.

This is not a policy or non-myopic result. Opportunity, development, and
holdout values remain sealed.

## Execution

- Preregistered implementation commit: `cb39437`
- Run ID: `voi-medical-dynamic-support-serving-20260726T061930Z`
- Model: `openai/gpt-5.4`, nonreasoning
- Requests and HTTP attempts: `10 / 10`
- Retries, reasoning tokens, forced exits, forced-final requests: `0`
- Cost: `$0.03062`
- Public artifact SHA-256:
  `e7effe8977f40fa25e8b8d43f7862d664ea3ee0f991c89a93fc65567d484bb0d`
- Private raw SHA-256:
  `c491492eb112ef53e4934a43aa5faee44d60be085233afe56d817c3941d74e17`

## Gates

- five strict six-hypothesis sets: pass
- five strict four-question sets: pass
- unique free-form hypotheses: `27 / 30` (gate `>=20`)
- unique questions: `20 / 20` (gate `>=15`)
- frozen target-substring exclusions: exactly row `474`
- eligible initial target coverage: `2 / 4` (gate `>=2`)

The model received only each self-report and its own generated differential. It
did not receive the target, stored conversation, or released diagnosis list.

## Mechanism Opportunity

The two covered eligible cases are Gastroenteritis and Gastritis. The two
initially missing targets are Gastric ulcer and Cold. Their generated supports
remain medically plausible but omit the external diagnosis, creating a
non-saturated mechanics test:

- can a hypothetical answer cause the LLM to regenerate a support containing
  the missing truth;
- do roots differ in that recovery behavior;
- does future-aware scoring identify roots with better realized support
  recovery than immediate fixed-support EIG?

These questions require a new frozen tree protocol. This serving pass alone
does not answer them.

## Budget

- run cost: `$0.03062`
- remaining frozen research allowance: `$1.10112155`
- last authenticated live balance: `$34.146247594`
- balance above protected `$25`: `$9.146247594`
- OatML jobs: `0`
