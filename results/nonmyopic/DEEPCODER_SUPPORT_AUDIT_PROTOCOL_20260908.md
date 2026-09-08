# Independent-program support diagnostic

This is a prospective numerical-adequacy diagnostic motivated by the completed
finite-prior pilot, not a replacement opportunity endpoint or a rescue of its
failed gate. Freeze code and this protocol before drawing new program outputs.

Bind `DEEPCODER_ACTIVE_PILOT_20260908.json` SHA256
`84bdcb025e70d3bbbb3640c5e40f86ae271f005098468f3056637665ba9aa476`.
Reconstruct its four reference matrices, program lists and input lists exactly;
verify all existing hashes before opening independent draws for that panel.
This is identity replay only: do not rerun any policy, optimize any action,
increase reference particles or revise the previous result.

For each panel generate 128 independent programs using the SAME syntax prior
and interpreter, seed 5100000+1000*panel+i. Evaluate the same 40 inputs. Keep
duplicates, constants and errors. All eight original query columns are probed
separately from the initial reference prior; there is no selected-query or
multi-step policy treatment. All 32 target columns remain fixed.

Report per panel:

- Unsupported exact query outcomes, by query and pooled. If an outcome has zero
  reference mass, weight-only Bayes conditioning is undefined. Count it; do not
  smooth, redraw, supply a truth particle, or substitute a default forecast.
- Unconditional prior half-Brier loss on independent targets versus the prior's
  own Bayes risk, and the number of actual targets assigned zero probability.
- On supported query conditions ONLY, held-out half-Brier loss versus internal
  posterior risk, target zero-mass count and surviving particle count. Label this
  selected subset explicitly. It is not an all-case policy utility estimate.

Any observed unsupported outcome establishes failure of this bank to provide
weight-only updates on that draw. Zero observed failures would not certify full
grammar coverage. There is no new efficacy threshold, paid authorization or
change to the old opportunity gate. Query conditions within a program are
dependent; no pooled IID confidence claim is made.

Reserve the new output path before source loading. Thirty seconds per panel,
four panels, finite 40960 bounded interpreter runs including identity replay;
checkpoint each panel. Unexpected errors bank failed_closed with no retry.
Save aggregate counts and independent-program/matrix hashes, not a new policy
comparison. No LLM calls, source benchmark dataset, cluster or automation action.
