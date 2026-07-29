# Number Game Qwen Pooled-Support Serving Preregistration

Date frozen: 2026-07-29, before seeds `63000` and `63001` are sent to a
model.

## Motivation

The item-isolated single-draw smoke was transport-clean but one conditioned
draw collapsed to two valid unique extensions. Its deployed retained support
still passed. This new method addresses support-sampling variance directly:
two independently seeded Qwen generations are pooled at every planning
history before posterior construction and lookahead.

This is not a threshold relaxation or rerun. It changes the estimator and
cost, uses fresh seeds, and leaves the failed single-draw gate unchanged.

## Exact-10 Mechanics Gate

- Model: `qwen/qwen3.7-plus`, nonreasoning.
- Two independent adapters, seeds `63000` and `63001`.
- Five fixed histories: one initial, two one-observation children, and their
  two linked two-observation children; two calls per history.
- Exact 10 accepted requests and HTTP attempts; zero retry/provider retry,
  reasoning, or forced exit; cost at most `$0.12`.
- Every live draw is strict JSON; no salvage is needed in the smoke.
- At each history the second draw contributes at least two new valid
  extensions after deduplication.
- Pooled initial support has at least 24 valid extensions; each conditioned
  pool at least eight; merged first and second supports at least 12 and eight.
- There is no minimum on either individual draw.

Synthetic tests verify independent request accounting, pooled encoding, and
extension-level union. Full pass alone authorizes a separately frozen fresh
32-tree cohort. Any miss closes this pooled Qwen route without reseeding,
threshold repair, or a third draw.

## Conditional Cohort

If authorized, use fresh tree seeds `63100..63131`, target seeds
`63200..63231`, eight single-draw cross-fit validation supports per tree from
`63300..63555`, and 20,000 bootstrap draws with seed `63600`.

Every planning history receives two independent Qwen generations; target and
validation draws remain single because the latter are already averaged over
eight independent supports. The retained-rejuvenation policy and exact
33-concept canonical endpoint remain unchanged.

Prospective primary gates:

- at least 28 depth-three-versus-myopic changed roots;
- changed-root realized advantage at least `0.008`, bootstrap lower above zero;
- simulated-to-realized Spearman at least `0.25`, bootstrap lower above zero;
- wins minus losses at least eight;
- policy Brier reduction versus myopic at least 8%, paired tree interval below
  zero, and at least 20 tree wins.

Mechanics require exactly 3,424 accepted requests, at most 24 transparent
provider retries, no reasoning/forced exits, at most 16 item-salvaged draws,
all pooled and retained support minima, a `$5.25` cap, and starting
provider-visible balance at least `$5.50`. Depth-two and other controls are
diagnostic. This fresh cohort cannot alter any previous run status.
