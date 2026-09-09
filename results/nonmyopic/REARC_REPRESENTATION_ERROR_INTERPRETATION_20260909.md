# Why executable Python did not supply usable planning branches

Saved-only audit: REARC_REPRESENTATION_ERROR_AUDIT_20260909.json.
It binds the completed result and all inspected artifacts, executes no candidate,
opens no new labels and makes no model call. Four focused synthetic tests passed.
The closed qualification result and all its gates remain unchanged.

## The Brier improvement is mostly reduced concentration

For normalized whole-grid Brier, DSL minus Python decomposes exactly into
`p_python(truth)-p_dsl(truth)` plus
`(sum p_dsl(outcome)^2-sum p_python(outcome)^2)/2`.

Mean gain0.214964096 decomposes into0.040555556 from truth probability and
0.174408541 from concentration reduction. The latter is81.13% of the gain.
This is a proper-score algebraic decomposition, not a causal attribution.
It explains why broad mixtures of wrong grids can improve scores while still
excluding the observations needed by a rollout tree. It does not erase the
measured cell-level improvement or the limited exact-output recovery.

## Not primarily an off-by-one problem

Best error among positive-weight, same-shape Python predictions on the12queries:

| Task | Query0 wrong cells | Query1 wrong cells |
|---|---:|---:|
|855e0971|0/180|527/720|
|4258a5f9|164/266|318/529|
|bd4472b8|78/90|147/161|
|be94b721|No supported correct shape|No supported correct shape|
|bc1d5164|No supported correct shape|No supported correct shape|
|868de0fa|44/396|56/625|

Thus four queries have zero posterior probability on the correct shape, seven
other incorrect queries miss44..527cells even under this oracle-best supported
comparison, and one query is covered exactly. These are not one-cell near misses.
This descriptive best-of-support diagnostic is not a deployable predictor and
does not change the metric to cell-level success.

## A concrete memorization route

All eight repaired Python slots for bc1d5164 contain a literal equal to the
observed demonstration output; together they receive posterior mass1.0. The
first repaired slot explicitly tests height9, width13, colors{1,9}, and count28
of color1, then returns that stored5x6output. Its remaining branch performs a
generic fixed-size downsampling. Exact observed fit consequently does not imply
the transformation was learned. The literal detector is intentionally only a
flag: presence alone cannot prove which branch executes, and absence cannot
prove lack of memorization. No candidate was rerun for this audit.

The shared verbal plans also fix demonstration-specific constants: task0 uses
color9, task1 color5 and background2, task3 a6x6 color4/9mask, and task5 fill7.
These may be valid competing hypotheses after one demonstration, but the
source varies instances. Constant-heavy plans plus exact single-example fitting
can yield a diverse, well-executed, badly unsupported predictive pool. This is
an inference from saved plans/programs and errors, not a controlled estimate of
the effect of any particular prompt phrase.

## Best next intervention

Do not increase depth, draw more branches from this unsupported pool, or soften
the exact-output gate. The next prospective experiment should test whether an
actual second observation induces genuine transferable program revision.
Use native Python in both arms, with a matched history-blind update control,
equal model calls/slots, common initial hypotheses and paired fresh targets.
One arm receives the newly revealed example; the control does not. Compare
predictive support and proper scores after the update, not observed fit alone.
Keep the mechanism-level plan and program compilation stages explicit so the
audit can distinguish semantic revision from patches that memorize the example.

This directly tests the LLM-native belief-update dependency required by the full
goal. It does not yet test non-myopia: a later prospectively frozen study must
show useful outcome-conditioned branch prediction, simulation/real-update
agreement and an actual terminal-risk horizon gap against compute-matched
myopic/random controls. If a second observation still cannot produce adequate
fresh-input support, deeper lookahead remains unjustified. Freeze exact cohort,
source qualification, sample size, call schedule and thresholds before new calls;
this document is a design decision, not paid authorization or a new success gate.

Previous turn: completed paid comparison (progress). Current turn: diagnostic
evidence changes the next action (progress). Account unchanged at usage
221.917614359/balance23.082385641; conservative London-day remaining3.50066412.
No API spend. Full non-myopic goal remains unmet.
