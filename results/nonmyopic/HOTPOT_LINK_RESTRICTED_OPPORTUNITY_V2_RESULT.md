# HotpotQA Link-Restricted Opportunity V2 Result

## Decision

The zero-call V2 structural screen **fails one conjunctive gate**:
mechanics contains `2` qualifying tasks rather than the frozen minimum `3`.

Development contains `24` qualifying tasks, exceeding the required `20`, and
every qualifying mechanics and development task has the intended unique
enabling-root optimum. Nevertheless, the exact gate does not pass, so no
model protocol or paid run is authorized.

## Results

- Mechanics rows materialized: `100`
- Development rows materialized: `500`
- Qualifying mechanics tasks: `2`
- Qualifying development tasks: `24`
- Malformed mechanics / development rows: `0 / 1`
- Candidate-count range per root:
  - mechanics: `0--1`
  - development: `0--3`
- Every qualifying task has one unique enabling optimum with support value
  `2`
- Model calls / cost: `0 / $0`

The first 20 qualifying development IDs were deterministically computed and
hashed, but they are not authorized for model use because the overall
opportunity gate failed.

## Integrity

- Mechanics split SHA-256:
  `ebaf587774b60728859c6c6d01a652f5ca5875e9edd1c6b299d6cd13611ea012`
- Development split SHA-256:
  `ad210c9e438c326a0d279631b733e5cc2b3af1e7121cbe69b30b173b425d7fc2`
- Selected-development ID SHA-256:
  `e158910288d6bede87136a90532c8dd5b7804ce2d1bcec47f438c0efe4105af0`
- Confirmation endpoint rows materialized: `0 / 2,000`
- Retained-holdout endpoint rows materialized: `0 / 68,791`

## Outcome

Close exact link-restricted V2 without relaxing the mechanics threshold,
moving development rows into mechanics, or sampling new cohorts until one
passes. The result banks strong structural evidence that the corrected action
graph restores a unique non-myopic target, but it is not an LLM policy result.

Confirmation and retained holdout remain sealed.
