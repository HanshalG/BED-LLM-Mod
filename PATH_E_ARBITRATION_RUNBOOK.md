# Path E Paprika Arbitration Headline Runbook

Status: **FROZEN BEFORE LAUNCH**

Date frozen: 2026-07-11

This runbook implements the scale-up authorized in `STATE.md`. Results must not change
the design, task range, recovery rule, endpoint, analysis, or claim rule below.

## Claim

The surviving Path E method claim is:

> Belief-guided one-step EIG arbitration over native thinking-LLM proposals improves
> interactive troubleshooting.

Faithful one-step BED/EIG and full two-step remain negative pilot results. Lookahead is
closed for this cycle. The headline does not rerun or rehabilitate those methods.

## Frozen Sample

- Benchmark: official hash-verified Paprika customer-service release.
- Split: `eval`.
- Tasks: the next 50 unseen tasks in released order, offsets 10 through 59 inclusive.
- Rounds: 5.
- Seed: 1304 for task/RNG/provider pairing.
- Censoring: unresolved tasks receive 6 censored turns.
- Pilot tasks 0--9 are excluded from every headline estimate.

The power artifact is `results/path_e/arbitration_headline/POWER.json`. Pilot paired
censored-turn deltas had mean -1.2 and SD 1.549. At half the pilot effect (0.6 turns),
the two-sided normal approximation requires 53 tasks for 80% power. The authorized
maximum N=50 is frozen, giving approximately 78.2% power.

## Frozen Arms

The thinking triplet runs in this exact order within every task shard:

1. `NaivePrimaryArbitration`: three public-history native proposals; candidate 0 is the
   native default; override only when the best alternative's EIG gap exceeds the frozen
   combined one-SE threshold.
2. `NaivePrimaryCandidate0`: belief-free prompt-matched control; generate the identical
   ordered proposal set and always execute candidate 0.
3. `naive`: existing 8k-thinking native one-action baseline.

The context arm is the existing non-thinking `naive` policy in separate 10-task blocks.
Generation-thinking EIG and full two-step are not headline arms.

**Pre-registered best-N amendment.** The development-only candidate-elicitation probe
was specified in `STATE.md` before held-out outcomes were opened. Its competitive read
requires a separate best-N EIG arm on the same held-out tasks 10--59. This arm requests
the five best next actions for resolving the issue quickly and applies ordinary
one-step EIG argmax. It is contextual: it does not alter the arbitration primary or
co-primary claim rule. Run it before the frozen headline analyzer, include its endpoint
in the manual disagreement audit, and report comparisons with thinking naive and
arbitration. Canonical recovery uses one complete EIG task shard per offset.

The same questioner model object and shared prompt cache are used for the thinking
triplet. Whenever arbitration and candidate 0 have identical public histories, their
ordered three-proposal sets must be byte-for-byte identical after parsing. The frozen
analyzer checks every such state. Any mismatch invalidates the causal comparison.

The method order charges arbitration for its own proposal generation. Candidate-0 raw
cost can be reduced by cache hits while histories match, so it is not interpreted as a
standalone deployment cost. Report arbitration cost, total coupled-evaluation cost, and
non-thinking/thinking-naive costs; label candidate-0 raw cost as cache-amortized.

## Models And Endpoint

- Questioner/belief model: OpenRouter `google/gemma-4-26b-a4b-it`, thinking enabled,
  8192 thinking tokens and 1024 final tokens for the thinking triplet.
- Simulator, mapper, likelihood, filter, and success judges: same model route,
  non-thinking.
- Non-thinking baseline questioner: same route, non-thinking.
- Native Paprika complete-conversation success protocol plus the committed strict
  terminal-faithfulness check.
- Answer-space coverage, structured failures, simulator contradictions, terminal
  checks/rejections, forced exits, requests, tokens, and costs are mandatory logs.

No arm may use the private solution for proposal generation, hypotheses, likelihoods,
or action selection. It enters only the released simulator and endpoint checks.

## Endpoints And Analysis

Primary endpoint: paired censored turns-to-resolution, arbitration minus thinking naive.

Co-primary causal endpoint: paired censored turns-to-resolution, arbitration minus the
prompt-matched candidate-0 control.

For both comparisons report:

- mean paired delta with a 5000-resample task bootstrap CI, seed 1304;
- turn wins/losses/ties;
- two-sided Wilcoxon signed-rank p-value as supporting analysis;
- resolution@5 delta, discordant resolution wins/losses, and an exact two-sided paired
  binary test.

Secondary/context results: full resolution curves, arbitration versus non-thinking
naive, answer-set coverage, forced exits, cost per resolution, request/token totals,
and total coupled-evaluation cost.

Claim B is confirmed only if:

1. automated and manual endpoint checks pass;
2. candidate pairing has zero mismatches; and
3. both primary and co-primary censored-turn 95% bootstrap CIs exclude zero in
   arbitration's favor.

Pre-registered weaker reads are:

- primary CI excludes zero and co-primary is directionally favorable: native gain
  supported, causal EIG gain uncertain;
- both are directional but either CI crosses zero: directional but uncertain;
- arbitration beats thinking naive but not candidate 0: proposal elicitation, not EIG
  override, explains the gain;
- no primary directional gain: Claim B not confirmed.

No threshold tuning, task subgroup selection, alternate censoring, or one-sided
replacement test is permitted after results are visible.

## Mechanism Analysis

Descriptive only, with no threshold changes:

- override rate;
- immediate resolution after an override;
- every override's score gap, one-SE threshold, and excess margin;
- override counts and mean margins grouped by arbitration win/loss/tie versus candidate
  0.

## Manual Endpoint Audit

After automated analysis, review:

1. every task where any of the five arms disagree on success; and
2. ten additional tasks sampled without replacement from the remaining tasks using
   NumPy RNG seed 1304.

Check each transcript against the private remedy, including every terminal success and
every attempted near-remedy. A correct performed remedy claimed to fail, or an
incorrect remedy claimed to succeed, invalidates the headline. Do not drop individual
tasks or arms. Quarantine the complete headline and stop-and-discuss.

## Canonical Recovery Rule

- Thinking triplet unit: one task and all three ordered methods in one invocation.
- If any thinking-triplet method fails, discard every artifact from that invocation and
  rerun the complete three-method task shard unchanged. Never combine partial methods
  across attempts.
- Non-thinking unit: one fixed 10-task block. If a block fails, discard the block and
  rerun the exact block unchanged.
- The first fully completed invocation for each unit is canonical. Failed attempts stay
  diagnostic and are recorded in `EXPERIMENTS.md`.
- Recoveries cannot change seed, prompts, thinking budget, retries, task membership,
  method order, threshold, or concurrency based on observed outcomes.

## Budget And Concurrency

Cumulative spend before launch is $10.38197. Hanshal added $10 during wave 1, raising
the authorized total from $20 to $30. Pilot rates project about $2.67 nominal for all
50 tasks/four arms; the conservative 2x envelope is $5.34. Pause for a budget top-up,
without inspecting policy results, if projected remaining work would cross $30. Notify
Hanshal if cumulative spend approaches $28.

- Thinking triplet: ten isolated task shards per wave, concurrency 25 each, aggregate
  ceiling 250. Five waves cover offsets 10--59.
- Non-thinking baseline: five 10-task blocks, concurrency 51 each, aggregate ceiling
  255. Do not overlap this wave with thinking shards.
- The user permits up to 256. Never increase a frozen shard's setting in flight.

## Frozen Configs

- `configs/config_paprika_arbitration_headline_triplet_openrouter.yaml`
- `configs/config_paprika_arbitration_headline_nonthinking_openrouter.yaml`
- `configs/config_paprika_best_n_eig_headline_openrouter.yaml`

Launch only from the pushed commit containing this runbook, configs, analyzer, control,
and passing focused tests. Record that commit and every launch in `EXPERIMENTS.md`.
Wave 1 was launched from the original frozen commit `7bc2263` with a $20 tracker cap.
The subsequent administrative amendment raises only the tracker cap to $30; it does not
change model, prompts, methods, tasks, seed, endpoint, analysis, or concurrency.

## Launch Pattern

Thinking task shard, offset `O`:

```bash
python -u main.py \
  -c configs/config_paprika_arbitration_headline_triplet_openrouter.yaml \
  --paprika-task-offset O \
  --run-name paprika-headline-triplet-oO-seed1304
```

Non-thinking block, offset `O` in `{10,20,30,40,50}`:

```bash
python -u main.py \
  -c configs/config_paprika_arbitration_headline_nonthinking_openrouter.yaml \
  --paprika-task-offset O \
  --run-name paprika-headline-nonthinking-bO-seed1304
```

## Frozen Combination And Analysis

Combine the 50 successful thinking shards:

```bash
python scripts/combine_paprika_step1_splits.py <50 canonical run dirs> \
  --methods NaivePrimaryArbitration NaivePrimaryCandidate0 naive \
  --expected-start 10 --expected-count 50 \
  --output-dir runs/paprika-headline-triplet-combined-seed1304
```

Combine the five non-thinking blocks:

```bash
python scripts/combine_paprika_step1_splits.py <5 canonical block run dirs> \
  --methods naive \
  --expected-start 10 --expected-count 50 --expected-shards 5 \
  --output-dir runs/paprika-headline-nonthinking-combined-seed1304
```

Analyze exactly once after all canonical artifacts are fixed:

```bash
python scripts/analyze_paprika_headline.py \
  --headline-run runs/paprika-headline-triplet-combined-seed1304 \
  --naive-nonthinking-run runs/paprika-headline-nonthinking-combined-seed1304 \
  --best-n-run runs/paprika-headline-best-n-combined-seed1304 \
  --expected-start 10 --expected-count 50 --round-budget 5 \
  --output results/path_e/arbitration_headline/PAPRIKA_HEADLINE.json
```

Do not launch MediQ until Claim B's automated result and required manual audit both
pass. If Claim B collapses, stop-and-discuss.
