# Active DeepCoder opportunity pilot

Prospective, new synthetic active-input task using ExeDec's DeepCoder interpreter,
not a reproduction of its released passive datasets or a reopen of relational
concepts. Freeze implementation and this protocol in git before pilot execution.

## Scientific contract

Public context is the interpreter vocabulary and sampling law only, identical
across hidden programs. Programs are not disclosed to a policy. Every experiment
returns the entire bounded integer/list output, or ERROR; no binning to manufacture
ambiguity. ERROR is a legitimate outcome, not an excluded program or input.
Unexpected implementation exceptions fail the panel. No source benchmark examples,
datasets, pretrained weights or held-out benchmark programs are downloaded.

The interpreter is Apache-2.0 ExeDec source, commit
`ef046ce2cc3fcd024e32f5dfe00e69700dac82ed`, file
`tasks/deepcoder/deepcoder_dsl.py`, SHA256
`b5564c6956ea1a49225c8e8d24b4a5118cda399ecfdadd7f2abca4e7960f4c42`.
Its bounded semantics use maximum integer magnitude 50. No generated Python is
executed; future proposals must use the typed DSL interpreter.

Our prior differs from ExeDec's outcome-filtered dataset sampler. Start with two
list inputs. Draw program length uniformly from 2,3,4. At each statement sample
uniformly among all type-valid (operation, lambda, argument) tuples in the full
source vocabulary. After the first statement, require use of the immediately
preceding result, preventing dead statements by construction. Earlier variables
remain available. This law depends on syntax/types only, never probe outcomes.
Keep duplicate draws, identical behaviors, constant functions and all-error cases.

Input law: two independent lists, each length uniform 1..5 and each element
uniform -10..10. This explicit narrower input range is new, not an upstream
default. Input and program seeds are independent. Four finite-prior panels each
contain 128 equally weighted program draws (seed 3100000+1000*panel+i) and 40
input draws (seed 4100000+1000*panel+j). The first 8 inputs are the action menu;
the remaining 32 are fixed, equally weighted predictive targets. Coincident
input values are retained; no policy-specific target deletion.

Primary loss is half multiclass Brier loss for the exact returned output,
including ERROR. Its Bayes risk is (1-sum_y p(y)^2)/2 averaged across the fixed
targets. This measures output prediction, not recovery of program syntax.
The prior-average evaluation draws truth from the SAME empirical prior. It is
not certification over the whole grammar or evidence of LLM proposal quality.

## Complete comparison and gates

Use the existing ordinary HorizonPlanner, structures fixed, exact categorical
conditioning. All arms have budget four, eight available queries, no repeats,
and deterministic action-index tie breaks. Compute full-budget h4 reference
first, then deployed receding h1/h2/h3, receding open-loop h3, uniform random
remaining queries, and exact-myopic control (identical to h1 here). Save initial
policy trees/root values, source identities and matrix/input/program hashes.

All four panels and all controls must complete. Success requires mean h2 <=
.95*mean h1, mean h3 <= .95*mean h2, positive h1/h2 risks, h3 strictly better
than receding open-loop (margin 1e-12), and at least 3/4 panels nonworsening in
depth (floating tolerance 1e-12 per adjacent comparison). Report full-budget
headroom regardless of success; two 5% gains need 9.75% total. No gate or prior
changes after outputs. Failed cases cannot be replaced.

Per planning call: five seconds, 100000 nodes, cache4096, depth<=4. Per complete
panel:120 seconds during planning; finite construction is 5120 bounded program
runs. Overall execution is supervised and must terminate; failures are banked,
not retried. Reserve the output path before loading the interpreter or executing
pilot programs. Checkpoint each completed panel. No transition database.

This stage costs zero model calls. A pass authorizes only prospective design of
a proposal-quality gate against history-blind and productive enumerative search,
not buying calls or claiming LLM-native efficacy. Use the real-history-only
refresh architecture first if those later gates pass. Anticipatory discovery
and powered confirmation remain separate unmet requirements of the full goal.

Sources: [ExeDec](https://github.com/google-deepmind/exedec),
[DeepCoder utilities](https://github.com/microsoft/DeepCoder-Utils).
