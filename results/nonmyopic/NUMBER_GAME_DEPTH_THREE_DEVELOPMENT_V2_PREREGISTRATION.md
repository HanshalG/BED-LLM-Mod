# Number Game Depth-Three Development V2 Preregistration

Date frozen: 2026-07-28, before any V2 development response.

V1 stopped after four complete trees on a paid malformed normal-stop response;
no partial endpoint aggregate was inspected. Strict parameter routing was
unavailable for Gemini. V2 uses entirely fresh seeds and swaps model roles.

One excluded GPT-5.4 Mini planner / Gemini target mechanics smoke used seed
`27591/27691`: 50/50 accepted responses, zero retries/reasoning/forced exits,
22 initial rules, first-branch minimum 9, second-branch minimum 4, 23 targets,
14 novel targets, and cost `$0.16479565`. Depth-three and depth-two roots were
both 36. Target endpoint metrics were not inspected.

## Frozen V2 Design

- Eight fresh GPT-5.4 Mini planning trees, seeds `27600..27607`.
- Eight fresh Gemini 2.5 Flash target supports, seeds `27700..27707`.
- The prompt-only hard executable-constraint repair, nonreasoning setting,
  temperature `0.7`, strict grammar, 50-call tree, twice-refreshed support,
  policies, controls, and exact endpoint are unchanged from frozen V1.
- Structural minima remain 16 initial, eight first-step, four second-step,
  16 targets, and eight novel targets.
- Exactly 400 accepted responses; only explicit zero-cost provider-error
  responses may receive the existing identical-payload retry.
- Total cost cap: `$1.50`; no reserve.

## Unchanged Promotion Rule

All mechanics must pass. Versus depth-two predictive risk, depth three must:

1. differ on at least 4/8 first roots;
2. improve aggregate Brier by at least 3%;
3. win at least 5/8 tree means on Brier;
4. avoid mean Hamming regression; and
5. avoid mean exact-extension coverage regression.

It must also have directional aggregate Brier gains over myopic EIG and exact
uniform random. Failure closes this depth-three implementation without branch,
tree, or threshold repair.
