# Synthetic compositional correction qualification

This is a capability gate, not a new LLM-native BED environment or headline. Source
laws are deliberately simple, numerical baselines are runnable, and query points
are fixed rather than selected by a policy. Do not claim irreducibility or depth.
No closed PhysGym or ChemBench output is rerun or used as an endpoint here.

Four fixed sources in order, x0,x1 positive in [.5,2]:
1. base=x0/(1+x0), source=base*exp(-.7*x1).
2. base=sqrt(x0), source=base+.4*x1**2.
3. base=x0, source=base/(1+x1**2).
4. base=sqrt(x0), source=base+.6*sin(x1).
The source formulas never enter the model prompt. Public descriptions only say
primary and secondary input of a synthetic response, not the missing mechanism.
Old history: (.6,1),(1.2,1),(1.8,1). New history additionally has (.8,.5),
(1.4,1.4),(1.1,2). All observations are log(source)+N(0,.05**2), noise seeds
59100000+i. Thirty-two target inputs are independently log-uniform on [.5,2]^2,
input seeds59200000+i; target noise consumes the next32 values after the6 history
noise values. Target labels evaluated only after all calls and forecasts seal.

Exactly8 Luna medium requests: control then refresh on even cases, refresh then
control on odd cases. Paired seed59300000+i. Model receives same base and public
feedback, first3observations for control and6for refresh. Both final numerical
predictors merge base plus decoded corrections and use all6 observations. Each
correction has same add/multiply tree format, width1-8,128nodes/depth32, existing
safe interpreter and domain checks. No cross-arm outputs or true correction in
prompts. Strict full-response decoder; transport/schema/resource failure stops,
no retry/salvage/alternative seed. All four cases retained in any completed score.

Numerical control is ridge regression of log-response on fixed features
[1,log(x0),log(x1),x0,x1,x0²,x0*x1,x1²], penalty.01, intercept unpenalized, all6
observations. Also report calibrated base. LLM predictors use existing exact
global log-scale prior N(0,4) and mixture likelihood, not fitted target weights.
These tests of missing shape are not a claim that the restricted posterior is
calibrated over an open world.

Gate: all final arms defined, refreshed mean target MSE <=90% of control mean,
at least2/4 paired improvements >.01, and refreshed mean nonworse than ridge.
Report every loss, pool, guard/history diagnostic and paired difference even on
null. No significance claim from4cases. Pass only motivates separately frozen
fresh-source transfer; independent joint answer/update fidelity and actual planning
headroom still required before depth. A null closes this exact qualification.

Full block cap$.32, $.04 reservation before each HTTP attempt, account-wide London
$5 daily cap. Existing authenticated route/receipt/budget lifecycle unchanged.
Freeze tested code before calls, save immutable forecasts before target labels,
independently replay source histories/targets, requests, bindings and receipts.
