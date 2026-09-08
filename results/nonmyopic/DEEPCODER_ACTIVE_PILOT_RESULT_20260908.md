# Active DeepCoder: complete finite-reference pilot

## Disposition

The prospective pilot frozen at `447697ec` completed all four panels and all
controls. Its joint opportunity gate is **null**. Do not change this pilot's
seeds, input law, program prior, target loss or thresholds to recover a pass.
No LLM proposal or paid policy study is authorized by these results.

This is a new active-input experiment using ExeDec's interpreter, not the
released ExeDec evaluation. The numerical reference is exact categorical
conditioning/search evaluated in floating point over an empirical program
prior, not a certified integration of the entire generative program grammar.

## Results

Mean terminal half-multiclass Brier loss after budget four, lower is better:

| Panel | h1 | h2 | h3 | Receding open-loop h3 | Full-budget optimum |
|---|---:|---:|---:|---:|---:|
| 0 | .00553385 | .00553385 | .00553385 | .00553385 | .00553385 |
| 1 | .00817871 | .00817871 | .00817871 | .00976562 | .00817871 |
| 2 | .00764296 | .00764296 | .00764296 | .00764296 | .00764296 |
| 3 | .00427246 | .00345285 | .00195313 | .00292969 | .00195313 |
| Mean | .00640700 | .00620209 | .00582716 | .00646803 | .00582716 |

Uniform-random mean risk is .01930162. Exact-myopic computation control equals
h1 by construction; it is not a substitute for a later productive-compute LLM
control. All means equally weight the four panels, with all prior draws retained.

- h1 to h2: **3.20%**, failing the frozen 5% threshold.
- h2 to h3: **6.05%**, passing that individual threshold.
- h3 versus receding open-loop h3: **9.91%** improvement.
- h3 versus h1: **9.05%** improvement, entirely concentrated in panel 3.
- All four panels are nonworsening with depth within declared numerical tolerance.

These are descriptive finite-reference effects, not a powered positive LLM
result. Full-budget reference equals h3 within floating precision in all panels;
its total headroom over h1 is only 9.05%, short of the 9.75% needed for two
successive 5% gains on this same population. A longer horizon cannot repair that
specific gate failure. In three panels h1 already reaches the full-budget optimum.

Initial risks are .44255--.46324, so nearly all initial predictive uncertainty
is eliminated even by h1 after four full-output queries. Different first-query
IDs alone would have overstated usefulness: roots differ in several tied panels.

## Retained cases and verification

Each panel has 128 prior draws and 40 fixed inputs. There are 127--128 distinct
program strings and 122--125 observed behavioral extensions per panel. All-error
and other constant extensions were kept (1--4 per panel). ERROR accounts for
17.56--29.61% of matrix entries, also retained. No difficult program or target
was removed. Full exact output categories were returned, not binary summaries.

The four panels completed in 1.069, .993, 1.152 and 1.079 seconds respectively
(4.293 seconds summed). The process exited normally. No SQLite or background
process remains from this pilot.

55 focused tests passed in .58 seconds, including the reused horizon planner,
an independent rational enumerator on constructed categorical fixtures, and
hash-pinned interpreter smokes. Scoped lint passed. A separate read-only check
of the saved artifact verified all 12 initial trees (2037 nodes): declared
depths, distinct actions on paths, positive normalized branch probabilities,
root minima, and branch-weighted risks. That check generated no new program
outcomes. It is structural/value replay, not a separate population replication.

Artifact: `DEEPCODER_ACTIVE_PILOT_20260908.json`, SHA256
`84bdcb025e70d3bbbb3640c5e40f86ae271f005098468f3056637665ba9aa476`.
The artifact includes program/input/matrix hashes and actual initial policy trees.

## Consequence for the full plan

We now have an inexpensive complete executable-program instrument and a specific
failure, rather than another unresolved deployment. This pilot does show some
useful anticipated adaptivity, but it does not support the desired robust ladder.
Its small finite prior is a serious scope limitation: full outputs can identify
a sampled program without identifying the actual generating program among the
much larger grammar. Do not generalize this headroom bound to the whole grammar.

The next scientific design must establish predictive-support adequacy alongside
headroom before using a small particle-bank oracle to select an environment.
Neither adding more LLM calls to this failed pilot nor selecting its one favorable
panel answers that issue. Any future broader-prior study needs a separate
prospective numerical-adequacy contract, with this null preserved and no claim
that changing the reference retrospectively rescues the original gate.

Model calls/spend: 0 / $0. Authenticated credits/usage remain
245 / 220.376693994, matching the London Sept8 ledger. No cluster use, protected
runtime changes or automation resumption. Full goal remains incomplete.
