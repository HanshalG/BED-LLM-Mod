# RegretBench Naive Baseline Independence Amendment

Date: 2026-08-07

**Status: prospective, implemented, unopened. Model calls: 0. Cost: $0.**

## Problem Found

The first naive-thinking amendment described Luna medium reasoning as a
descriptive baseline that could not pass, rescue, or veto the primary
DeepSeek experiment. The implementation did not fully enforce that boundary:

- the naive smoke had to pass before primary development opened;
- formal Luna request and action-coverage gates were included in the primary
  mechanics conjunction; and
- a Luna or naive-endpoint exception aborted the whole formal run.

Thus an unmatched descriptive baseline could have converted an otherwise valid
dynamic-versus-myopic result into `mechanics_failed`. No policy response,
hidden truth, or endpoint had been opened when this inconsistency was found.

## Prospective Fix

The enriched DeepSeek smoke remains the only policy smoke that can gate the
primary development experiment. The Luna naive smoke is still attempted and
banked, but failure disables only the descriptive baseline.

Formal execution now uses three adapters with one shared run ID and one hard
`$3.50` reservation boundary:

1. primary DeepSeek planning and realized policy histories;
2. optional Luna medium-reasoning naive questions; and
3. optional DeepSeek support measurements on the naive history.

Primary request, retry, reasoning, schema, privacy, and supported-action gates
use only adapter 1. Adapters 2 and 3 have separate descriptive diagnostics.
Their calls still count against combined spend, but their failures cannot
change primary mechanics, scientific gates, status, or authorization.

The primary maximum is `8,768` DeepSeek requests: exact `8,256` planning plus
at most `512` realized-history requests. An available baseline adds at most
`128` DeepSeek endpoint requests and `128` Luna requests, preserving the
original `9,024` combined maximum and `$3.50` formal cap.

## Failure Semantics

- Naive smoke failure: bank the smoke failure, skip all formal baseline calls,
  and execute primary development.
- Formal naive failure: bank the stage and error, omit `naive_thinking` from
  descriptive comparisons, and complete the primary result.
- Naive action coverage below `48/40`: report it descriptively; do not alter
  primary status.
- Any primary DeepSeek failure: retain the original fail-closed behavior.

The first Luna question is still frozen before hidden truth access whenever
the baseline is available. No response, endpoint, seed, prompt, policy,
scientific threshold, task split, draw count, or budget was changed.

## Verification

The exact-scale synthetic path now covers all `8,256` planning responses plus
realized policy histories in three cases:

- baseline available;
- formal Luna call fails deliberately; and
- baseline disabled after smoke failure.

In both failure cases, primary mechanics pass and the preregistered scientific
summary is computed. Daily orchestration also proves that a failed naive smoke
continues into development without constructing either formal baseline
adapter. The combined source/support/policy/budget suite passes `64/64`.

This amendment is validity repair before data, not scientific evidence.
