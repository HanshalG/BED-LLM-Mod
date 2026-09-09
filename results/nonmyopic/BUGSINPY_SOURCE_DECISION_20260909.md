# BugsInPy: useful source, not yet a BED environment

Audited upstream revision `11c5f1eea954a42132cfd06bf257766a7963e0fd`.
Only README, Dockerfile, checkout/test framework, and recursive path metadata were
read. No bug descriptions, patches, task programs, or test outcomes were opened;
no upstream code ran. Hashes and paths are in BUGSINPY_CONTRACT_AUDIT_20260909.json.
The metadata inventory contains 501 bug records across 17 projects. No root license
file was found; this is not a conclusion about individual project licenses.

## What the source actually supplies

- Buggy and fixed revisions, plus a single-test selector. This is more promising
  for executable hypotheses than a prose-only localization benchmark.
- Checkout copies fixed-version tests into the buggy checkout and copies bug
  metadata containing revision identifiers. Exposing the resulting directory to
  an LLM would not implement sealed outcomes. A separate public workspace is needed.
- The relevant-test driver classifies output containing `passed` as successful.
  A literal translation accepts `1 failed, 3 passed in 0.10s`; the regression
  reproduces this without running an upstream test. Do not use this predicate for
  observations or terminal scores. Single-test mode also prints raw output.
- Dockerfile uses a mutable base-image tag and downloads the latest uv installer.
  Presence of a Dockerfile is not verified reproducibility or a candidate-code
  isolation guarantee. No build was attempted.

Sources: [pinned test driver](https://github.com/soarsmu/BugsInPy/blob/11c5f1eea954a42132cfd06bf257766a7963e0fd/framework/bin/bugsinpy-test),
[checkout](https://github.com/soarsmu/BugsInPy/blob/11c5f1eea954a42132cfd06bf257766a7963e0fd/framework/bin/bugsinpy-checkout),
[Dockerfile](https://github.com/soarsmu/BugsInPy/blob/11c5f1eea954a42132cfd06bf257766a7963e0fd/Dockerfile).

## Scientific distinction that must survive implementation

Running a known buggy program is not by itself an unknown-world experiment: its
behavior is already determined by visible source. Candidate patches differ in
intended behavior, not necessarily their predictions of that known program.
Counting debugger steps as BED depth would therefore misstate the claim.

A coherent formulation would hide the intended executable specification and let
the agent purchase selected input/output observations from it. LLM-generated
repairs would then be executable candidate worlds; unseen input/output prediction
would be a proper terminal target. But BugsInPy's fixed revision is only a proxy
for intended behavior, not automatically an exhaustive specification. Exposing
assertions, fixed patches, commit identifiers, or unrestricted git history would
defeat that uncertainty. This differs from the closed SWE-bench retrieval route;
it cannot inherit that route's results or bypass its null by renaming it.

## Decision and next dependency

Keep BugsInPy as a candidate source, not a selected benchmark or a measured null.
Do not build a general repair agent, launch paid calls, or claim a horizon gap.
Before an adapter, prespecify a small metadata-selected development sample of
deterministic public APIs, then inspect public pre-fix specifications and project
licenses only. Require callable finite input domains, stable reference behavior,
isolated execution feasibility, and a public/candidate/endpoint split that can
withhold assertions without removing the task's semantic context. Ineligible
cases must remain recorded, not be replaced after outcomes.

If that source contract passes, freeze the actual intended-behavior query task and
test LLM proposal coverage and observation-conditioned predictive improvement
against same-history redraw and a productive numerical/executable control. Only
then measure independent rollout transition fidelity and an ordinary fixed-budget
d1/d2/d3 gap with compute-matched myopic and random controls. No artificial unlock
or monotonicity-by-policy-improvement substitution. Source eligibility alone is
not evidence for any of these scientific properties.

Validation: 2 tests passed in .09s using the existing Anaconda interpreter. System
Python lacked pytest; no dependency install was needed. Zero model calls/cost.
Authenticated usage221.306531939, balance23.693468061; London Sept9 conservative
spend .88825346 including prior .04 uncertainty, remaining4.11174654.
Previous turn was no progress (status restatement); this turn banks new pinned
source evidence and a reproduced observation-classification hazard. Goal remains
active and unachieved.
