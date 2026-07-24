# PAPRIKA Murder-Belief Opportunity Preregistration

Date: 2026-07-24
Seed: `24329`
Status: preregistered before any response.

## Scientific question

Do free-form evidence-unlocking investigations create realized two-step value in
an LLM's own path-dependent suspect belief?

This is the first gate after closing single-chain MuSiQue. Each PAPRIKA mystery
has several plausible suspects and witnesses under one public scene. A private
world specifies the culprit, motives, evidence, and character behavior. The
released environment explicitly allows actions to unlock items and additional
questions, so the first observation can create a useful continuation rather than
merely reveal one member of a commutative document pair.

## Frozen data and models

- Released PAPRIKA `murder_mystery.json`, SHA-256
  `2ae865e60662135f6e867a95653be48ecc8b6a51cddc5c3e46e2538ae24b2184`.
- Target-blind sample without replacement from 50 eval cases, seed `24329`.
- Smoke indices: `38, 47`.
- Opportunity indices: `31, 27, 46, 39, 12, 18`.
- Reserve only: `19, 44, 43, 25, 20, 32`.
- GPT-5.4 non-reasoning, temperature 0: suspect beliefs and actions.
- GPT-5.4 Mini non-reasoning, temperature 0: official PAPRIKA environment and
  final semantic equivalence.
- The investigator sees only the public scene and observed responses.
- Only the environment sees the private culprit/evidence scenario.
- The truth reference is parsed from the explicit released culprit clause and is
  shown only to the final equivalence judge.

## Frozen tree

The initial investigator call returns:

- exactly eight culprit hypotheses with probabilities summing to one;
- three direct suspect-discriminating actions;
- three evidence-unlocking investigations.

The six actions are shuffled. The official PAPRIKA environment answers each one
independently from the hidden world. After every response, GPT-5.4 regenerates
the eight-suspect belief and proposes four distinct response-conditioned
follow-ups. The environment answers all 24 follow-ups with the corresponding
first-turn history, and GPT-5.4 regenerates a final belief. The first complete
history is replayed once through the belief generator.

Each case therefore uses exactly 63 physical calls:

- 1 initial belief/action call;
- 6 first environment responses;
- 6 first belief/follow-up calls;
- 24 second environment responses;
- 24 final beliefs;
- 1 final-belief replay;
- 1 batched equivalence judgment.

Private raw responses are checkpointed after every phase. Public artifacts retain
actions, response hashes, and measured truth probabilities.

## Frozen gates

Smoke requires exactly 126 calls, zero reasoning, valid eight-hypothesis beliefs,
finite replay values, and at least three distinct first-response hashes per case.

The six-case opportunity conjunction requires:

1. exactly 378 calls and zero reasoning;
2. mean initial truth probability at most `.35`;
3. mean distinct first responses at least `4.0`;
4. one-step truth-probability spread at least `.10` on at least 4/6 cases;
5. two-step oracle first action differs from realized greedy on at least 2/6;
6. pair gain over best one-step probability at least `.10` on at least 3/6 and
   mean at least `.10`;
7. oracle gain over the realized-greedy continuation at least `.10` on at least
   2/6 and mean at least `.07`;
8. replay truth-probability gap mean at most `.10` and maximum at most `.25`.

This is an opportunity gate, not a policy result. Passage would authorize a
separately preregistered target-blind planner/ranker gate on reserve cases.
Failure closes this exact PAPRIKA interface with no subset, prompt, model,
threshold, or response repair.

## Budget

- Smoke projected/hard cap: `$0.75` / `$2.00`.
- Opportunity projected/hard cap: `$2.25` / `$6.00`.
- Pre-smoke project headroom is `$36.816935534` above the protected `$25`
  Monday reserve. OatML remains paused.
