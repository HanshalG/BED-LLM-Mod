# Headroom first; targeted causal mechanisms next

Previous turn was progress: exact decomposition established a root-rank reversal
and a finite-population ceiling on all further planning improvements. This turn
implements the corresponding necessary-condition check and reassesses the next
direction against primary literature. No new outcomes or paid experiments.

## Executable Necessary Condition

For positive myopic loss R1 and required relative adjacent gains g1,...,gk:

    required final loss <= R1 * product(1-gi).

Two successive5% gains therefore need9.75% total improvement, not merely5%.
If a certified LOWER bound L on optimal full-budget risk is larger than that
ceiling, no policy can meet the ladder on that same population/loss/budget.
An achieved policy risk is an UPPER bound on optimal risk and cannot be used to
rule out headroom unless its optimality is independently established.

`core/research_headroom.py` implements this with exact rationals and explicit
scope. Its only conclusions are `ruled_out` and `not_ruled_out`. The latter is
not success, a proof of monotonicity, or permission to spend. It does not validate
a caller's mathematical certificate or automatically modify existing runners.
Thirteen tests passed0.22s; lint passed. Invalid/approximate inputs, zero baseline,
boundary equality and conservative bounds are covered.

Applied read-only to the two SHA-verified relational result files:
86b5ad18... and dbadd7a9..., the requested total gain is39/400 and maximum possible
gain is4.0424%. Result `ruled_out`, with no recomputation of labels or policies.
All old statuses/gates are unchanged. A first local invocation lacked NumPy for
the repo's core package imports; application succeeded under the declared runtime.

## Primary Literature Recheck

[GO-CBED, Zhang et al., v1](https://arxiv.org/html/2507.07359v1)
optimizes sequential intervention policies for specified causal quantities rather
than the entire model. It combines variational posterior estimation with an
amortized policy. Its limitations include dependence on priors over structures
and mechanisms; dynamic model updates remain an extension. This motivates
targeted causal mechanisms as a candidate, NOT importing a demonstrated LLM
depth ladder. Goal alignment and non-myopic planning must be ablated separately.
The paper's motivating fixed-graph comparison alone does not isolate horizon
benefit. No verified runnable upstream release was established in this pass.

[Kandasamy et al., ICML2019](https://proceedings.mlr.press/v97/kandasamy19a.html)
analyzes myopic posterior sampling using conditions related to adaptive
submodularity. It is a warning against assuming a complex hypothesis language
automatically creates a large greedy-policy deficit. It does not prove our
relational task is adaptively submodular or explain its measured small gap by
itself; that would require checking the actual task's utility and assumptions.

[Murphy, MDA v4](https://arxiv.org/html/2608.09696v4), AppendixD.5,
reports unstable source-finding design estimates around near-singular signal
regions and no advantage of active design over random there. This is consistent
with taking our historical location estimator failures seriously, but is not a
reproduction of our configurations. MDA's executable-proposer/numerical-inference
division remains useful. Its v4 prompt-leak correction was already recorded in
our earlier plan; it is not new progress or permission to reuse older prompts.

## First-Principles Design Requirement

The next candidate should have independently motivated complementary experiments:
one observation calibrates a nuisance mechanism that determines what another
observation can teach about a fixed target. This is stronger than adding more
formulas, increasing inference noise, or introducing an arbitrary query unlock.

For intuition only, independent fair bits U,Z and observations U and U XOR Z
have no individual information about Z but reveal Z jointly. That is a simple
mathematical example of complementarity, NOT an environment proposal, LLM result,
or proof of adaptive branching value. A fixed pair already solves it. A credible
task additionally needs outcome-dependent choice among later useful experiments
and measured advantage over a precommitted sequence/receding open-loop control.

Our inference from the literature and the accumulated local nulls is to prioritize
goal-oriented causal mechanism inference with scientifically justified nuisance
uncertainty. Keep LLMs as evidence-responsive executable mechanism proposers;
numerical code must own likelihoods, weights and control evaluation. The target
must be chosen for scientific meaning before outcomes, not selected from targets
whose oracle gaps happen to be large. Classical structure/parameter proposals get
the same source information and measured resource limits.

## Bounded Next Decision

One source-grounded contract check for the GO-CBED causal-mechanism direction:
identify exact reproducible model/prior/intervention/target definitions and
whether a bounded complete full-budget reference is feasible. No broad benchmark
sweep, no another random grammar, no silent reuse of closed ChemBench or CA-BED
endpoints. Treat any new discretization or altered observation model as a new
task, not numerical acceleration of the original one.

Before writing an environment runner, explicitly identify the LLM's useful
inference role and the symbolic baseline. A small known DAG/function library
cannot establish discovery merely because the LLM can name its entries. If the
source gives only that, it is a supporting reference rather than a headline.

Only after a credible source/model contract may a fresh frozen full-budget
headroom screen open. Its first question is whether the entire requested ladder
is even possible. A non-rejection still requires actual h1/h2/h3, branching,
deployable-prior accuracy and useful proposal gates. This direction is a research
choice, not evidence those gates will pass. No source execution or paid call is
authorized by this note alone.

Account refreshed245/220.376693994/24.623306006; London Sept8 spend0,
remaining$5. No cluster work, automation paused, full goal unfinished.
