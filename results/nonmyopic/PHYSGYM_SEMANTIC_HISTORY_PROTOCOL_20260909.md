# Prospective four-task semantic/history proposal gate

## Scope fixed before responses

This is noisy source-function identification on mathematically code-valid inputs,
NOT a claim about physically validated experiments or a faithful unchanged PhysGym
benchmark. Selected103/458/457/653 stay fixed. No additional tasks, outcome-based
replacement, independent replicas represented as independent tasks, or gate rescue.
The mathematical expressions and their source-validity checks define the oracle;
source Python and hidden-equation correctness judges never execute in the policy.

Each positive real input, including dummy inputs, is sampled independently
log-uniform on[.5,2]; N is uniform integer3..12. These domains are declared numerical
test domains, not physical ranges. Observations are log(source output)+Normal(0,.05²),
with evaluator-owned randomness. Use four observed points and32 independent targets
per task; seeds53100000+i for inputs and54100000+i for observation/target noise.
No source-output rejection; invalid output fails that task and remains in report.
All source functions must be positive/finite on sampled inputs in zero-call preflight.

## Paired interfaces

Use exact openai/gpt-5.6-luna medium reasoning, same established provider route and
16384token ceiling, up to8 scalar expressions/request. Sixteen requests total:
initial semantic and initial blind on the same first3 observations, then refreshed
semantic on4 observations and semantic redraw on the original3. Initial arms
alternate order by task index, as do the second pair. Pair seeds55100000+10*i and
55100001+10*i. Semantic context includes source problem text and variable meanings;
blind omits them. Both use anonymous x0.. variables and identical numeric history.
Never send ID, dataset name, equation, answer, solution or implementation.

For the later comparison BOTH predictors condition numerically on all4 real
observations. Pool initial-semantic proposals with the refreshed or redrawn pool,
dedupe exact AST syntax, then weight by the same fixed log-Gaussian likelihood.
Thus the tested difference is proposal construction given the new observation,
not withholding that observation from the control's Bayesian update. The earlier
semantic/blind comparison uses their own pools and3observations. The symbolic
baseline is intercept+log-input ridge regression(.01 non-intercept penalty), all
variables including dummies, fit on4observations. No target labels enter fitting.

Invalid/negative/nonfinite formula behavior on public history or target INPUTS is
excluded and counted; empty support is a task failure, never silently omitted.
Forecasts include weighted mean log response. They are restricted proposal beliefs,
not a calibrated full posterior merely because weights use a likelihood.

## Endpoints and gates

Seal all16responses/pools/forecasts before evaluating any target outputs. Primary
endpoint: mean squared error for the independent noisy log responses, a proper
point-prediction score. Report all4tasks, aggregate losses and raw paired differences.
Empty support or failed request closes the complete gate; do not assign a favorable
finite surrogate score or drop the task. Context gate: semantic initial mean MSE
at least10% below blind and at least2/4 task wins>.01. History gate: refreshed mean
MSE at least10% below redraw, at least2/4 task wins>.01, and mean MSE no worse than
the symbolic baseline. All four must have nonempty support in every arm. No claims
of statistical significance or non-myopic efficacy from this four-task development
screen. Passing permits only a fresh joint transition/planning fidelity design.

## Cost and execution prerequisites

Hard full-block worst-case cap$.64, reserve$.04 per HTTP attempt. Daily account-wide
$5 cap, existing uncertainty retained, exact prices and credits reread per attempt.
No retries, no fallback providers, no model escalation within this protocol. Any
request error banks terminal failure. Needs tested fail-closed runner, response
schema enforcement and independent replay before actual dispatch. This written
protocol alone does not launch or authorize an unqualified runner. No planner,
cluster or background automation is introduced.
