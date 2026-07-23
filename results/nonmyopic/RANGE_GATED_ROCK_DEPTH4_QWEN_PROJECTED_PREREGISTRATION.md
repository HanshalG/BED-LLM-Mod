# Range-Gated Rock Depth-Four Qwen Bounded-Projection Preregistration

Status: frozen after deterministic producer/audit tests, before any live response
under the projected interface.

## Motivation and Prior Boundary

The all-or-nothing Qwen seed-24212 smoke is closed. Its registered correction
response contained the exact critical `move-NORTH, move-NORTH, move-NORTH,
check-4` plan, but illegal tails under two other fixed roots caused whole-cell
rejection. No plan was scored.

This is a new serving-mechanism test on fresh cells. It never rescues or
retrospectively scores seed 24212. The prompt, model, K4 roots, exact EIG scorer,
scientific controls, and thresholds remain unchanged. Only invalid-branch
compilation changes.

## Frozen Projection Rule

Each response gets the same single correction attempt. Branches that are
dynamically legal in either response are retained exactly, preferring the latest
valid model branch. A branch still missing or illegal after correction is
compiled to:

```text
fixed-root, check-0, check-0, check-0
```

This projection is legal from every grid position and contains no movement after
the fixed root. It therefore cannot create the critical three-move route or any
on-site travel continuation. Every branch records its exact source
(`attempt_0`, `attempt_1`, or `projected`).

The line may pass only when the exact selected h4 plan is fully LLM-authored.
Projection is serving support for non-selected comparison branches, not a source
of planning quality.

## S0: Projected Serving and Contribution Smoke

- Model: dense Qwen 3 14B thinking.
- Fresh seed `24216`.
- Ten distinct cells.
- 4,096-token reasoning allowance, bounded 256-token finalization, temperature
  zero, and one correction attempt.
- Hard OpenRouter run cap `$1`.

S0 passes only if:

1. all ten cells compile to four legal fixed-root plans;
2. the exact `N,N,N,check-4` route is present in at least `8/10` cells;
3. exact scoring selects the exhaustive-h4 root in at least `8/10` cells;
4. every selected plan is sourced from a Qwen response, never projection;
5. every cell retains at least one Qwen-authored branch and no more than 75% of
   all branches are projected;
6. no projected branch equals the critical route; and
7. all serving usage is retained and exact scoring makes no model calls.

Any failed condition stops this projected Qwen line without repair.

## Conditional S1: Proposal Quality

S1 runs only after S0 passes.

- Fresh seed `24217`.
- Sixteen distinct strict h4-over-h3 opportunity cells.
- Producer bootstrap seed `24218`; 5,000 paired replicates.
- Independent audit bootstrap seed `24219`.

Controls remain:

1. identical-root random h4 tails;
2. shared-plan h3 scoring;
3. strongest exact d3 root with its best exhaustive h4 continuation; and
4. exhaustive h4 for route and opportunity recovery.

Both producer and independent audit must have strictly positive paired 95% lower
bounds against the first three controls, at least 75% exact complete-route
selection, at least 60% mean opportunity recovery, and all S0 contribution
mechanics. The audit must additionally recover every LLM branch from its
serialized attempt and independently regenerate every projection.

A pass is bounded-projection LLM-Modulo h4 proposal evidence. It is not unaided
LLM planning and does not itself constitute a trajectory result.
