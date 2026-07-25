# ClariQ Dynamic-Support V2 One-Call Serving Gate

## Purpose

Test the sole V2 transport change on fresh disclosed mechanics topic `60`
without generating branches or loading endpoints. V1 topic `38` is not rerun,
and its response is not reused.

The V2 manifest SHA is
`fa5a34e55ab455359a4a64bd2aba00ea5789f27fb2f2d932ea9e2330cf90ca03`.

## Frozen Call

- Model: `openai/gpt-5.4`, OpenRouter, non-reasoning.
- Temperature `.7`, maximum 1,024 output tokens.
- Exactly one initial-support request.
- Exactly eight lines:
  `Hxx|mass|A B C ...|short intent`.
- Codes must have exactly one ASCII space between them, one valid code for each
  of the 15 questions, and no leading/trailing/double spaces.
- Nonnegative unnormalized masses are accepted prospectively if total mass is
  positive.
- Raw response is checkpointed before strict parsing.
- No cleanup, parser normalization, response repair/reissue, branch call,
  endpoint load, development access, or holdout access.

## Conjunctive Gates

- exactly one physical request and HTTP attempt;
- zero retries, reasoning tokens, and forced exits;
- all eight support lines parse;
- at least four distinct positive-mass response profiles;
- support entropy at least `1.0` nat;
- initial myopic-EIG range across 15 roots at least `.05` nats;
- zero branch requests and no endpoint load; and
- cost at most `$0.05`.

Passing authorizes only a separately implemented and preregistered fresh V2
full mechanics tree on topic `60`. Failure closes dynamic-support V2 without
another format or model attempt.

## Budget

Projected cost is `$0.02`, with a hard `$0.05` cap. The stricter project ledger
has `$13.091824290776685` remaining before this call; the `$25` Monday reserve
remains protected. OpenRouter only; no OatML/cluster use.
