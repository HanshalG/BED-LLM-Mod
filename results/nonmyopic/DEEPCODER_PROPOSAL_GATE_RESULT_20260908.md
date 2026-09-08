# Proposal-only gate: source-grammar interface failure

Protocol and runner were frozen and pushed at 0a20c236 before execution.
Run directory: `deepcoder_proposal_gate_20260908/`.
The one-shot gate terminated on its FIRST model response. No retries, repairs,
replacement cases or extra model calls were made. Target outcomes, predictive
comparison and depth experiments did not open. This is not a predictive-quality
null, an LLM-versus-search result or evidence about planning horizon.

## What happened

- Exact model: deepseek/deepseek-v4-flash-0731; provider: OpenInference.
- Explicit nonreasoning request; response reported zero reasoning tokens.
- Clean stop, 863 prompt tokens, 167 completion tokens; valid JSON.
- Four proposed programs; all four violate the source-prior requirement that
  each statement after the first consumes its immediately preceding result.
- For example, the first program sorts x0 into x2, then takes Head of x1 rather
  than consuming x2. This is a type-valid ordinary program but is OUTSIDE the
  frozen source grammar. The gate correctly refuses to treat it as a source-prior
  candidate. This distinction matters: it did not return arbitrary broken Python.
- Exact response SHA256:
  289095b6019d3875aa9e72cbd1b6ff5515c1a02dde715b7bcd560e5268c41224.

Independent zero-call replay confirmed serving validity and each of the four
same grammar violations. `diagnosis.json` records this without repairing or
executing counterfactual corrected programs. The rest of the LLM panel was never
requested. Prior to that request, all64 bounded symbolic restarts completed,
with41 compatible fits and54,867 operation attempts. Those counts alone do not
establish predictive performance or a comparison with the LLM.

## Cost and accounting

One accepted call cost $0.00006987, well below the $0.25 block cap. Authenticated
account totals after execution still showed credits245 and usage220.376693994,
so posted usage lagged the accepted response. The ledger therefore records the
larger locally measured cost, leaving $4.99993013 in the London daily allowance.
No outstanding reservation remains. The authorization is consumed by this
terminal attempt; unspent allowance does not authorize a rerun or another test.

## Interpretation and next dependency

The immediate failure is interface adherence, not tested reasoning intelligence,
predictive accuracy or sequential BED. A single response cannot estimate the
model's failure frequency. No efficacy conclusion follows either way.

Our current interface asks the model to obey a structural restriction that a
decoder could enforce mechanically. A genuinely new successor should represent
only type-valid, previous-result-consuming programs, with that property guaranteed
by construction rather than repaired after generation. It would need its own
prospective interface/source-law checks, controls and explicit spending scope.
Do not retry this prompt, enlarge its sample or relax its validation after the
response. Even a successor grammar pass would leave predictive utility and
non-myopic opportunity unproved.

47 focused tests passed in1.85s before launch, including complete synthetic
16-call execution, forecast sealing, endpoint bombs, abstention handling,
single-attempt transport failure and cost reconciliation. All durable artifacts
are banked. No cluster use or automation change; the full goal remains incomplete.
