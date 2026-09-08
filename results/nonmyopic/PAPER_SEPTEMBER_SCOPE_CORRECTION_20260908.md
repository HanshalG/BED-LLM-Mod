# Paper scope reconciled with September evidence

Prior turn made empirical progress: validation-only interpolation transferred a
small first-step predictive gain in the saved bank. This turn updates the working
paper instead of letting that diagnostic become an unqualified headline.

Changes to paper/main.tex and paper/README.md:

- Historical Number Game depth labels explicitly mean root selection with greedy
  future queries, not full receding-horizon optimization.
- Positive correlation between simulated and realized root advantage is no longer
  described as full predictive calibration of regenerated beliefs.
- Finite symbolic controls no longer support a blanket assertion that classical
  methods cannot discover useful rules or model support transitions.
- The different fixed-101-target exact-horizon audit, its failed 5% second-link
  gate, uniform-refresh degradation and 0.78% validation-weighting improvement are
  reported together, with retrospective/analyst-examined-target qualifications.
  The offline weighting result is not attributed to the new online learner.

Added three hash-bound diagnostic bundles to paper/claim_manifest.json. All 44
bundles pass hash/value validation. This validates artifact identity and selected
values; it does not establish that every scientific claim is true or publishable.
Fourteen focused paper-validator/scope tests pass (0.35s). The full LaTeX/BibTeX
validation passes all required limitations, both figures and the six-page limit.

The first edited build was seven pages. A build of the unchanged pre-edit source
verified the six-page baseline; redundant narrative/ranking detail was condensed,
without changing fonts, margins, thresholds or historical statuses. Final direct
render is six pages. First/last pages were visually inspected for the affected
abstract, qualifications and reference overflow; no clipping/overlap was seen on
those pages. This is not a new full publication-readiness review of all figures.

Generated test build is local scratch, not a replacement for archived frozen
manuscripts or terminal handoffs. No archived experiment/paper variant is rerendered
or reclassified, no endpoint opens, and no paid call occurs. The full goal remains
unachieved; a fresh controlled proposer-plus-weighting and non-myopic policy study
is still missing. Automation remains paused.
