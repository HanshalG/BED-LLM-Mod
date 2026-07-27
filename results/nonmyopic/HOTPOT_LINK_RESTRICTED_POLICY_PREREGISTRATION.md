# HotpotQA Link-Restricted Policy Development Preregistration

## Status And Scope

This is a **method-development** experiment on already-open rows, not a
confirmatory claim.

The prior structural gate failed because its mechanics slice had two rather
than three qualifying rows. It nevertheless opened mechanics 100 and
development 500, found 24 qualifying development tasks, and verified that
every qualifying task has a unique enabling-root optimum under paragraph-link
actions. V4's five rows are also open for diagnosis.

No confirmation or retained-holdout endpoint has been opened. This protocol
uses the first qualifying mechanics row for serving and the already-frozen
first 20 qualifying development rows for one model-development measurement.
Only a conjunctive pass can authorize a separately frozen confirmation.

## Corrected Environment

After a root paragraph is revealed, available second actions are:

- `STOP`; and
- context titles literally mentioned in that paragraph.

No unmentioned title can be selected. Frozen `strict_unlock` guarantees that
the enabling paragraph links to the answer article and the answer paragraph
does not link back, making the enabling root uniquely optimal.

GPT-5.4 nonreasoning generates eight open hypotheses, refreshes them after
each of four root paragraphs, and ranks complete paths. Ranking outputs use
strict numeric-root rows:

- myopic: `Rn|rank`;
- aligned/fixed/shuffled: `Rn|action|rank`.

Myopic chooses from the initial belief but receives the aligned continuation
for its root. Thus its second-link selector is compute-matched and the primary
comparison isolates first-link choice.

## Calls And Controls

Exactly ten logical calls per task:

1. initial eight-hypothesis belief;
2. four root-conditioned refreshes;
3. myopic root rank;
4. aligned complete-path rank;
5. fixed-initial-belief complete-path rank;
6. cyclic-shuffled-belief complete-path rank; and
7. final answer from non-myopic selected evidence.

Controls are myopic receding, fixed belief, shuffled belief, and seeded random
receding. Supporting facts and answer endpoints remain hidden until all model
outputs and policy choices freeze. Bounded logged transport retries are
allowed; semantic repair or reissue is forbidden.

## Serving Gate

One already-open qualifying mechanics row, exact ten calls. Require all
parsers, four changed and distinct beliefs, exact request accounting, zero
reasoning/forced exits, and cost at most `$0.25`.

## Development Gate

If serving passes, run all 20 frozen development tasks exactly once. Every
condition is conjunctive:

- all 80 refreshes change and every task has four distinct states;
- aligned rank differs from fixed and shuffled on at least five tasks;
- non-myopic changes at least four myopic roots;
- non-myopic selects at least 12 enabling roots and at least three more than
  myopic;
- non-myopic covers at least `32/40` support documents;
- versus myopic: gain at least `3`, wins minus losses at least `3`, at most two
  losses, and exact one-sided sign-flip `p <= .05`;
- versus fixed: gain at least `2`, wins minus losses at least `2`, and at most
  three losses;
- gain at least `4` versus shuffled and `8` versus random;
- exact structural root-ranking accuracy at least `.70`, at least `+.05` over
  myopic, and at least `+.05` over fixed;
- mean final-answer token F1 at least `.50`; and
- cost at most `$3.00`.

The fixed-belief comparison is essential: a pass must show that correctly
path-conditioned LLM belief regeneration contributes beyond merely seeing the
link action graph.

Failure closes this development interface and leaves confirmation sealed.
Pass authorizes only a separately frozen powered confirmation.

## Budget

- Serving projected/cap: `$0.15 / $0.25`.
- Development projected/cap: `$1.50 / $3.00`.
- Authenticated balance: `$29.287059594`.
- No fixed reserve.
- OpenRouter only; no OatML or Slurm.
