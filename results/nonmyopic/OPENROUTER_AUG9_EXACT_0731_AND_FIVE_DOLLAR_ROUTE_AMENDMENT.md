# OpenRouter Exact 0731 And Five-Dollar Route Amendment

Date: 2026-08-09 (Europe/London)

Status: zero-call budget and model-routing amendment. This supersedes only the
benchmark-availability caveat in
`OPENROUTER_AUG9_DAILY_BUDGET_AND_FRONTIER_AMENDMENT.md`. It does not change any
frozen model, prompt, effort, task, seed, endpoint, gate, request count, or paid
authorization.

## Live Account Boundary

After sourcing `.env`, authenticated OpenRouter reads still report:

- total credits: `$245.000000000`;
- cumulative usage: `$220.121013787`;
- available balance: `$24.878986213`;
- current-key daily usage: `$0.000000000`.

The newly reported `$30` is not yet visible in `/api/v1/credits`, so it is not
counted. The current balance already covers the frozen Aug 10 mechanics allowance
and all four Aug 11--14 development-plus-baseline allowances: `$2.00 + 4 * $4.95 =
$21.80`, leaving `$3.078986213`. If the top-up posts without intervening use, the
balance becomes `$54.878986213`; after the entire currently possible mechanics,
development, and conditionally authorized confirmation schedule, the remaining
balance would be `$14.078986213`.

The hard account-wide cap remains `$5.00` per London calendar day. Unrelated use
counts. A preregistered gate, transport failure, or finite stage may correctly end
a day below the cap. No filler calls are authorized.

## Exact Current Model Evidence

Artificial Analysis now identifies its `/models/deepseek-v4-flash` page as the
July 31 `DeepSeek V4 Flash 0731` revision. At max reasoning it reports:

- Intelligence Index `52`;
- `$72.03` total cost for the complete Intelligence Index evaluation;
- `210M` output tokens and `127.8` output tokens/second;
- text-only input, a `1M` context, and `284B` total / `13B` active parameters.

The current Luna xhigh page reports Index `50`, `$95.13` full-evaluation cost,
`67M` output tokens, `184.8` output tokens/second, and text-plus-image input. These
generic measurements make DeepSeek 0731 the stronger price/intelligence choice for
hard text in this pair, while Luna remains the only applicable route for direct
Bongard image-conditioned beliefs.

The authenticated OpenRouter catalog is even more favorable to the same split:

| Exact route | Input / output per 1M | Input modalities |
|---|---:|---|
| `deepseek/deepseek-v4-flash-0731` | `$0.09 / $0.18` | text |
| `openai/gpt-5.6-luna` | `$0.10 / $0.60` | text, image, file |

Direct project evidence remains the deployment gate. In the matched 128-request
nonreasoning Number Game support test, DeepSeek cost `$0.034477701` versus Luna's
`$0.057383`, making DeepSeek `39.92%` cheaper for that exact batch. DeepSeek parsed
`128/128` with no forced exits and produced mean conditioned support `12.108`; Luna
had four initial parse/forced exits but produced richer mean conditioned support
`15.925`. Both had a worst-case conditioned support of zero and both failed the
strict semantic-support gate. Therefore neither generic index nor clean JSON alone
authorizes a scientific stage.

## Frozen Routing And Daily Allocation

1. Keep Luna nonreasoning for the frozen Bongard image-conditioned belief dynamics.
2. Keep Luna medium for the separately labelled visual thinking/naive baseline.
3. Use exact dated DeepSeek 0731 nonreasoning for new bulk text-only support,
   likelihood, planner, and audit work only after an exact-interface semantic gate.
4. Use DeepSeek high, then max, for small hard text gates with fixed output and cost
   caps. Do not silently enable reasoning.
5. Preserve the frozen daily schedule: Aug 10 at most `$2.00`; Aug 11--14 at most
   `$4.95` per day; Aug 15--18 at most `$4.75` per day only if development literally
   authorizes confirmation.

Replacing Luna with DeepSeek inside the unopened visual protocol would change the
scientific interface and is forbidden. The exact 0731 benchmark changes future
model preference for text work, not the registered Bongard execution.

Sources:

- https://artificialanalysis.ai/models/deepseek-v4-flash
- https://artificialanalysis.ai/articles/deepseek-v4-flash-0731-scores-50-on-the-artificial-analysis-intelligence-index-10-points-above-previous-deepseek-v4-flash
- https://artificialanalysis.ai/models/gpt-5-6-luna-xhigh
- https://developers.openai.com/api/docs/models/gpt-5.6-luna
- https://api-docs.deepseek.com/quick_start/pricing/

This amendment made zero paid model calls and spent `$0.00`.
