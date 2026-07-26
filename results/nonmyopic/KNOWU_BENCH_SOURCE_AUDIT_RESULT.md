# KnowU-Bench Source Audit Result

Audited: 2026-07-26. This is a zero-call source gate.

## Pinned Source

- Official repository: `https://github.com/ZJU-REAL/KnowU-Bench`
- Commit: `c03a825991ede13add6631f2ed19b90755930dc6`
- Preference-task aggregate SHA256:
  `a67717f0f4f665e15e941625acfe21225481824282eba760d516df131b80f846`
- Four-profile aggregate SHA256:
  `d77840606416c3b94c3a2de294ff4564a30e5cfda537081f6427799c93712a3e`
- Four clean-log aggregate SHA256:
  `d5acd87e9d2b8528bc388b0dc7eea48363514de4a1c09a01d1bec83368f1d2d8`

## Why This Source Is Different

KnowU-Bench releases four structured user profiles (`developer`, `grandma`,
`student`, and `user`) plus profile-conditioned behavioral logs. Its official
task registry cross-products routine and preference tasks with those profiles.
The preference release contains:

- 26 task families;
- 86 official task/profile variants;
- 10 hard, interactive task families supporting at least three profiles;
- a free-form `ask_user` simulator conditioned on the hidden profile;
- full clarification history passed back to the simulator;
- task-specific programmatic or hybrid endpoints.

This is the first audited source in the current search where the same released
task is intentionally instantiated under several released hidden semantic
worlds. The task/profile cross-product is benchmark code, not a post-hoc world
fabrication.

## Limitation

The benchmark does not itself establish a non-myopic BED problem. It publishes
no profile prior or reference clarification policy, and unrestricted
`ask_user` permits a direct terminal-preference question. A depth effect under
that interface would be artificial.

## Authorized Derived Route

The source audit authorizes only a separately frozen dynamic-support
construction:

1. use official hard task families and supported released profiles;
2. hide profile labels and use a uniform prior;
3. expose only task-relevant clean behavioral logs initially;
4. permit one atomic preference question per turn and reject profile-label or
   compound terminal-preference questions;
5. let the LLM generate and regenerate semantic profile hypotheses from the
   realized history;
6. compare dynamic depth two against dynamic myopic, matched-compute myopic,
   and random controls;
7. evaluate truth-support recall, truth log probability, entropy AUC, and the
   official task-conditioned preference endpoint.

This must pass a content-sealed task-family manifest, an exact serving fixture,
and a capped mechanics gate before any efficacy run.

OpenRouter spend: `$0`. OatML use: none.
