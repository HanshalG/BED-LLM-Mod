# Bongard Luna Matched History-Blind Control Amendment

Date frozen: 2026-08-06, before any Bongard model request, candidate endpoint,
or scientific endpoint was accessed.

This amendment adds a direct causal control for answer-conditioned VLM belief
regeneration. It does not change the task split, model, visible images, query
budget, endpoint, primary dynamic-versus-myopic comparison, or confirmation
boundary.

## Motivation

The existing fixed-support arm asks whether analytical lookahead over one VLM
support is sufficient. The shuffled-continuation arm preserves the distribution
of generated continuation values while breaking their coupling to first
actions. Neither directly asks whether placing the simulated answer in the VLM
prompt improves planning beyond making another semantic generation call.

That distinction is central to the LLM-native claim. A matched history-blind
arm is therefore required before any Bongard response is generated.

## Paired Branch Generations

For every task, first candidate, and simulated binary label, issue two branch
requests with the same 14 opaque images and the same request seed:

1. the dynamic request contains the initial four labels plus the simulated
   candidate label;
2. the history-blind request contains only the initial four labels.

The two prompts differ only in the simulated answer. Candidate and endpoint
roles remain evaluator-private. The history-blind request does not contain a
candidate ID, label, branch marker, draw index, or semantic nonce beyond the
ordinary opaque image list already present in every request. Distinct branch
pairs receive distinct deterministic seeds; the dynamic and history-blind
member of one pair share a seed as a common-random-number control. Provider
seed compliance is not assumed and is reported only as the requested design.

Each history-blind response supplies a fresh ten-rule support, initial-history
weights, and image likelihoods. For a simulated label, the evaluator applies
one analytical Bernoulli update to those weights, excludes the queried image,
and scores the best remaining candidate. The frozen control score is

```text
history_blind_score(a) = root_EIG(a)
  + P_root(+|a) max_{b != a} EIG(blind[a,+] updated by (a,+), b)
  + P_root(-|a) max_{b != a} EIG(blind[a,-] updated by (a,-), b).
```

This differs from fixed support because every branch slot receives a fresh
semantic draw. It differs from dynamic support only because the draw did not
see the simulated answer.

## Execution Contract

Add `history_blind_depth2` as a sixth policy. After it selects a first action,
execution uses that action's real answer-conditioned branch and the same
branch-greedy second query and final history regeneration as dynamic, myopic,
and shuffled policies. Thus policy endpoint differences arise from first-action
selection, not a weaker deployed updater.

All root, conditioned-branch, history-blind-branch, and final requests are
strictly seed-manifested and replayed. The full mechanics tree now has exactly
4 root, 64 conditioned-branch, 64 history-blind-branch, and 16--40 final
requests: 148--172 total. Each eight-task development block has exactly 264
first-stage requests and 32--80 final requests: 296--344 total. Existing
`$1.75` mechanics and `$4.75` daily development caps remain unchanged; observed
serving cost per request must project the enlarged maximum tree under the
existing 1.5 safety multiplier before authorization.

## Mechanics Gates

In addition to every existing gate, mechanics requires:

- exactly 64 history-blind branch responses with the initial history only;
- exact paired request-seed and prompt-difference accounting;
- finite history-blind scores and an exact branch map;
- at least one non-tied dynamic-versus-history-blind first-action change across
  the four mechanics tasks.

The last condition ensures the direct control is behaviorally identifiable
before development opens. No four-task endpoint direction is required.

## Development Gates

The original dynamic-versus-myopic Brier endpoint remains primary. The matched
history-blind comparison is a co-required mechanism gate for authorizing a
confirmation preregistration. Across the frozen 32 development tasks require:

1. dynamic and history-blind final histories differ on at least 12 tasks and
   in every execution block;
2. dynamic improves mean endpoint Brier by at least 3% relative to
   history-blind;
3. the paired bootstrap probability of a Brier improvement is at least 0.80;
4. dynamic mean log loss is no worse than history-blind;
5. dynamic mean root-score ranking fidelity is no worse than history-blind.

All paired means, sample standard deviations, intervals, and wins/ties/losses
are reported even when the conjunction fails. A failure is a development null;
it cannot be removed after endpoint access.

Mechanics and development interfaces advance to `-4`. The development
manifest and the August 10 wrapper binding must be regenerated before any paid
request. Earlier unopened interface artifacts remain invalid.
