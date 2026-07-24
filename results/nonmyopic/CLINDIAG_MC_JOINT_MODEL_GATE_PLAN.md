# ClinDiag Multiple-Choice Joint-Model Gate

Date: 2026-07-24

Status: **preregistered before any response from this interface.**

## Motivation

The native-block ClinDiag gate failed because one coarse block usually saturated
support value. The direct test-card replacement was rejected statically because a
retrospective procedure menu is selected under the true case and omits outcomes for
unperformed tests.

This gate tests a materially different interface:

- a target-blind model proposes specific clinical questions or tests;
- every query has exactly four mutually exclusive outcome options;
- a hidden case gatekeeper selects the realized option from the complete ClinDiag
  record, synthesizing a case-consistent result when the record is silent;
- a likelihood model estimates `p(y | history, hypothesis, query)` over the same
  enumerable options.

This follows BED-LLM's prior-likelihood construction and small answer space while using
SDBench's case-conditioned gatekeeper pattern. It is not yet a policy or horizon test.

## Frozen Smoke

- Selection seed: `24291`.
- Four fresh cases, disjoint from every prior ClinDiag split and the sealed holdout:
  - `22704279`;
  - `13424741`;
  - `rare126`;
  - `rare82`.
- Models:
  - initial support and query generation: non-reasoning GPT-5.4;
  - hidden gatekeeper, likelihood, and audit judge: non-reasoning GPT-5.4 Mini.
- Initial support: 12 open-world diagnoses from only the public initial presentation.
- Query set: exactly four requests per case, each with outcomes `A` through `D`.
- The query generator sees the initial presentation and generated support, never the
  hidden record or target.
- The gatekeeper sees the hidden record and target but must return only an outcome ID,
  an objective finding, and whether the finding was recorded or synthesized. It may
  not diagnose, interpret, or hint.
- The likelihood model sees the public history, six generated hypotheses, the true
  hypothesis added only after queries and realized outcomes are frozen, and the four
  query option sets.
- Query 1 is sent twice with an exact identical gatekeeper prompt for each case.
- One independent audit call per case checks response relevance, objectivity,
  target leakage, case consistency, and duplicate semantic consistency.
- Temperature is zero except the already-qualified initial support sampler.
- No retries are allowed.

Expected physical calls:

| Stage | Calls |
|---|---:|
| Initial supports | 4 |
| Candidate query sets | 4 |
| Original gatekeeper outcomes | 16 |
| Exact gatekeeper duplicates | 4 |
| Likelihood matrices | 4 |
| Independent response audits | 4 |
| **Total** | **36** |

## Frozen Gates

All conditions must pass:

1. exactly 36 physical calls and zero reasoning tokens;
2. every support has 12 diagnoses;
3. every case has four valid unique queries and four valid options per query;
4. all 16 original gatekeeper outputs parse and pass relevance, objectivity,
   no-target-leak, and case-consistency checks;
5. all four exact duplicates select the same outcome ID and pass semantic consistency;
6. all likelihood rows parse, contain finite nonnegative probabilities, and sum to one;
7. mean probability of the realized outcome under the true hypothesis is at least
   `0.35`;
8. the true-hypothesis realized-outcome probability exceeds the mean of the six
   generated hypotheses on at least 8 of 16 queries;
9. the mean true-minus-generated realized-outcome probability margin is at least
   `+0.05`.

The final two criteria are diagnostic likelihood-fidelity screens. Because the true
hypothesis is injected only after the target-blind actions and outcomes are frozen,
they do not leak into action construction.

## Stop Rule

Failure closes this exact prompt/model interface. No threshold repair, structural
lookahead gate, planner, or holdout follows. Passing authorizes only a fresh,
preregistered structural-opportunity gate with replicated belief refresh; it does not
establish a non-myopic benefit.
