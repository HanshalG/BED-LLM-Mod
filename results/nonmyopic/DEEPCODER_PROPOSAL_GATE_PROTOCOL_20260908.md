# Approved proposal-only DeepCoder screening gate

## Authorization and scope

Hanshal explicitly approved changing order: a new proposal-quality test may run
before full-grammar numerical-reference qualification, with TOTAL exposure at
most $0.25 inside the account-wide $5 London-day limit. No depth sweep or reopening
of an old task/model interface follows. This is an eight-problem engineering
screen, not powered efficacy evidence, calibrated Bayes or the completed goal.

## Frozen cases and interfaces

Use unchanged pinned ExeDec grammar/prior/input law from deepcoder_opportunity.py.
Eight fresh independent source programs have seeds 8100000+i for i=0..7. For each,
34 inputs have seeds 9100000+100*i+j, j=0..33. First two are actual observed
examples; remaining 32 are fixed predictive targets. No rejection, stratification,
constant/ERROR exclusion or case substitution. Targets are never deleted because
they duplicate observed inputs. No hidden program or seed enters model messages.
Public input histories are prepared before calls; target OUTCOMES are evaluated
only after all forecasts have been sealed and semantic coverage passes.

Use exact deepseek/deepseek-v4-flash-0731, nonreasoning explicitly, temperature .7,
4096 maximum completion tokens, seed 11100000+i shared by the aware/blind pair.
Existing strict proposal prompt and compiler are unchanged. Request JSON-object
format; perform strict whole-response executable validation locally. First call
is also serving/schema gate, not an additional calibration response. Any malformed
completion, truncation, nonzero reasoning, HTTP failure or missing cost stops
the entire gate without retries/repair/salvage, and opens no target outcomes.
Aware then blind on even cases; reverse order on odd cases. Exactly 16 calls
maximum, serial. Each proposes up to eight 2-4-statement source-valid programs.

History-blind gets identical vocabulary/prior/instructions and empty history.
Both arms are numerically filtered against the SAME actual history. No generated
weights. Uniform compatible syntax-deduplicated pool prediction, explicitly NOT
full source-prior inference. Keep source-prior helper and earlier results unchanged.

Symbolic control: eight CrossBeam restarts, each 2048 operation attempts, weight9,
5 seconds, shuffled operation order seeds 10100000+10*i+r for r=0..7. Collect at
most eight first-fitting expressions, deduplicate syntax and apply same history
filter/uniform prediction. Source search still merges observational equivalents
within a restart; reordered restarts may or may not yield diverse predictions.
Identity/shorter/longer tree solutions remain allowed, giving search broader
expressivity than the source prior. This is a bounded productive-search and
maximum-width control, NOT a token/FLOP/walltime matched baseline. Report its work.
No old search history is rerun or its cap increased to rescue its old result.

## Frozen disposition and metrics

All eight cases and three arms must exist in the sealed panel. Empty compatible
pools are explicit abstentions. If aware coverage is below 6/8, bank coverage-null
without opening targets. Otherwise score all 256 targets per arm. The primary
screening loss is abstention-adjusted half-multiclass Brier: ordinary predictive
Brier when a pool exists, fixed failure penalty 1 when it does not. This penalty
is an operational failure cost, NOT a probability distribution or a proper score
on abstaining cases. Report coverage and per-case losses so the distinction is
visible; never report surviving cases alone as overall accuracy. Report zero-mass
target counts, counting all targets of an abstention as unpredicted.

Screen passes only if ALL hold: aware coverage >=6/8; mean adjusted Brier at least
10% below EACH control; strictly better per-case loss in >=5/8 cases versus EACH;
no more zero-mass targets than either control. Ties are not wins. This is a
prospective engineering threshold, not a significance test. Passing authorizes
only designing the next calibration/planning prerequisite, never automatic depth
experiments or a publication claim. Failing closes this exact gate without new
seeds, prompt repair, cap expansion or post-hoc threshold changes.

## Accounting and failure handling

Refresh authenticated catalog before each request; require input price <=$0.10/M,
output <=$0.20/M and explicit nonreasoning support. Provider max_price enforces
these ceilings; provider fallbacks disabled and parameter support required.
Prompt content JSON <=16,000 UTF-8 bytes; reserve conservatively for up to65,536
input tokens and4096 output tokens, including ample framing margin. Reserve
$0.015 per HTTP attempt in durable ledger before dispatch;16 attempts reserve
at most$0.24, below the$0.25 authorization. Re-read live account usage immediately
before every attempt and count unrelated usage above the frozen day boundary.
No borrowing or top-up inference. Any uncertainty retains the reservation and
terminates. Accepted cost is reconciled even if content fails. No retries.

Exclusive run directory prevents reruns, and a ledger lock prevents this runner
from overlapping cooperating launches. External account users cannot be locked;
their posted usage is checked before every request. Do not run another local
paid block concurrently. Record raw requests/responses, costs, provider catalog,
public cases, symbolic work, forecast seal, terminal disposition and permitted
outcomes. Never record credentials. Verify zero-call synthetic full pass/null,
endpoint bombs, daily accounting and single-attempt uncertainty before paid use.

Provider contract sources checked before launch:
https://openrouter.ai/docs/guides/routing/provider-selection
https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
