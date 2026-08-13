# CausaLab LLM-Native Source Admission Result

Date: 2026-08-13

Status: **source failed closed; exact release closed as an LLM-native headline route.**

## Result

The new official CausaLab release is complete enough to evaluate as a source. It
contains 950 graph configurations across 19 suites and exposes a genuine sequential
experiment: a hidden SCM governs both an intervention crystal and a held-out reactor,
the agent has a finite experiment budget, and the final causal graph/equation is
machine parseable. Graph identity and environment randomness can also be bound for
local replay while hidden graph values remain outside the policy prompt.

The frozen audit therefore passed release, native-sequential-experiment, and
replay/sealing gates. It failed the next required gate, open-support ownership.

The official `OnlineInterventionCausalTool`:

1. reads every graph configuration in the active group;
2. constructs an executable SCM for every candidate;
3. infers candidate-specific base values from the observed state;
4. executes each observed intervention under each candidate; and
5. deterministically removes candidates whose simulated transition disagrees with
   the observation.

It can also put the surviving graph IDs and edges directly into the LLM prompt. Thus
the released task family already supplies a complete finite hypothesis bank and exact
source-owned transition evaluator. Omitting that summary from a prompt would make the
LLM approximate information already available to a classical enumerator; it would
not make semantic hypothesis generation load-bearing.

| Gate | Result |
|---|---|
| immutable source and nonempty release | pass |
| native sequential causal experiment | pass |
| replay and endpoint sealing | pass |
| open-support ownership by the LLM | **fail** |
| irreducible LLM role | not evaluated |
| zero-call horizon opportunity | not opened |

## Interpretation

CausaLab remains useful as an interactive-science benchmark and potentially as a
classical or hybrid BED baseline. This result does not test whether depth-two planning
helps on its SCMs, whether an LLM can recover the mechanism, or whether its endpoint
can be predicted. It says only that this exact release cannot establish the project's
strong claim that the LLM irreducibly owns an open, path-dependent belief process.

No graph record, bundled trajectory, intervention outcome, transfer endpoint, or model
response was opened. No environment was executed, and no OpenRouter or cluster call
was made. Per the preregistered stop rule, no mechanics or horizon-opportunity screen
is authorized.

## Integrity

- source protocol SHA-256:
  `41deda4b99dcbbc9cb60befcde9e94a975fec0b5bf98e6c86fd28cabf872fa7b`;
- aggregate source result SHA-256:
  `cad8aefd51d52b2afa74e4eb442e30ec3b1e10217b1cc2531e44e8f02ff6db27`;
- audit implementation SHA-256:
  `3452d3ab7dfce0365015b1d450d8a4737b243fe3aaafd6442965a7847ea1415e`;
- focused test SHA-256:
  `56f0f0443e49400ead001afb5a8a1a983f964db52655414fca6e2b0c43452db2`;
- CausaLab commit/tree:
  `42ba47fb88e60dc1eca17bd29c47ace4e8e9960e` /
  `5700be5d041eda64c56d610cdaa9dfaa1e7736c1`.
