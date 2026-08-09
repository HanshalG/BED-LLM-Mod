# OpenRouter Posted Credit And Budget-Model Route Correction

Date: 2026-08-09 (Europe/London)

Status: zero-call account, pricing, and model-routing correction. This supersedes
the exact-0731 benchmark numbers in
`OPENROUTER_AUG9_EXACT_0731_AND_FIVE_DOLLAR_ROUTE_AMENDMENT.md`. It changes no
frozen model, prompt, reasoning effort, task, seed, endpoint, gate, request count,
or paid-stage authorization.

## Posted balance and daily boundary

After exporting variables from `.env`, authenticated OpenRouter reads report:

- total credits: `$245.000000000`;
- cumulative usage: `$220.121013787`;
- available balance: `$24.878986213`.

The newly reported `$30` is not yet present in `/api/v1/credits`. It cannot be
spent or included in authorization until that endpoint changes. If it posts with
no intervening usage, the balance will be `$54.878986213`: ten complete `$5`
London-day ceilings plus `$4.878986213`.

The hard account-wide cap is `$5.00` per Europe/London calendar day. Every daily
runner must snapshot authenticated cumulative usage, count unrelated account use,
reserve the worst-case cost of every HTTP attempt before dispatch, and reconcile
against the larger of posted and locally recorded spend. There is no borrowing or
rollover. Aim to use the allowance on the strongest dependency-valid evidence, but
do not manufacture filler calls after a gate closes a branch.

The frozen schedule remains:

| Date or dependency | Strongest authorized work | Daily ceiling |
|---|---|---:|
| Aug 10 | serving plus finite Mechanics4 | `$2.00` |
| Aug 11--14, after an exact mechanics pass | Development16 plus paired Luna-medium naive baseline | `$4.95` per day |
| Aug 15--18, only after literal confirmation authorization | Confirmation24 | `$4.75` per day |

The finite Aug 10 stage is intentionally below `$5`; spending the difference on
an endpoint-informed improvised experiment would be worse budgeting, not better.
If a registered gate closes the Bongard branch, a new text-only route must first
freeze its own task, controls, endpoints, and semantic gate before using a later
day's allowance.

## Exact current model evidence

The user's `7031` refers to the dated `0731` revision. OpenRouter's authenticated
catalog currently exposes:

| Exact route | Input / output per 1M | Context | Input |
|---|---:|---:|---|
| `deepseek/deepseek-v4-flash-0731` | `$0.09 / $0.18` | `1,048,576` | text |
| `openai/gpt-5.6-luna` | `$0.10 / $0.60` | `1,050,000` | text, image, file |

At output-dominated list price, DeepSeek is about `3.33x` cheaper. A `$4.75`
science envelope corresponds to at most about `26.4M` DeepSeek or `7.9M` Luna
output tokens before prompts, retries, and attempt reservations; these are capacity
comparisons, not permission to issue calls.

Artificial Analysis's exact July 31 article reports DeepSeek V4 Flash 0731 at
Intelligence Index `50`, one point behind GPT-5.6 Luna max at `51`, with roughly
`60%` lower first-party cost per task. DeepSeek used about `206M` output tokens,
remains text-only, and has `284B` total / `13B` active parameters. This exact source
supersedes the earlier local note's `52` versus `50` figures, which mixed changing
indexed model-page data with the dated revision.

Direct project evidence remains the deployment gate. In the matched 128-request
Number Game test, DeepSeek cost `$0.034477701`, parsed `128/128`, and produced mean
and minimum conditioned support `12.108/0`; Luna cost `$0.057383`, had four initial
failures, and produced `15.925/0`. Both failed the strict semantic-support floor.
DeepSeek's price and generic score therefore do not authorize bulk scientific use
without an exact-interface semantic gate.

## Role-specific route

1. Use `deepseek/deepseek-v4-flash-0731` nonreasoning for new bulk text-only
   support, likelihood, planning, and audit work after exact schema and semantic
   gates.
2. Use DeepSeek high, then max, only for small hard-text gates with fixed output
   and cost caps. Never silently enable reasoning.
3. Keep `openai/gpt-5.6-luna` nonreasoning for frozen image-conditioned Bongard
   beliefs. DeepSeek cannot receive Bongard images.
4. Keep Luna medium for the separately labelled visual thinking/naive baseline.
   Escalate visual reasoning to Luna high before max only through a fresh gate.
5. Do not swap either model into an already frozen experiment after observing an
   endpoint. Task-normalized reliability and scientific role outrank generic AA.

Sources:

- https://artificialanalysis.ai/articles/deepseek-v4-flash-0731-scores-50-on-the-artificial-analysis-intelligence-index-10-points-above-previous-deepseek-v4-flash
- https://artificialanalysis.ai/articles/gpt-5-6-has-landed
- https://openrouter.ai/deepseek/deepseek-v4-flash-0731
- authenticated OpenRouter `/api/v1/models` and `/api/v1/credits` reads

This correction made zero paid model calls and spent `$0.00`.
