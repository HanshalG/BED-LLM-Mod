# Full-export grammar: expressive vocabulary, unusable uniform prefix

Pinned Hodel sourcec60faf57 compiles into a353-rule permissive Herb grammar:
160functions,28constants,189terminals includingI,160direct-call rules and4
computed-call forms. Argument/return annotations are retained as metadata, not
silently interpreted as a sound type system. All exports remain available,
including resizing, objects and higher-order operations. This is vocabulary
coverage, not a proof of bounded coverage of every valid128-step DAG.

The direct single-call Cartesian space alone contains1,372,880,124expressions
before filtering. This count includes many nonsensical combinations and is not
a count of valid programs. It excludes deeper expressions and computed calls.
Full bottom-up product materialization was therefore not attempted; the actual
bounded smoke uses Herb's lazy BFS iterator, max_depth5/max_size12, first256
candidates, no observational-equivalence pruning. This is not the previous
small-grammar cost-based experiment or a controlled comparison against it.

Actual enumeration completed in the existing pinned Julia1.10.10/Herb/Search
environment. Run isolation: read-only, uid65534, no network, no capabilities,
no-new-privileges,2GiB/2CPU,128process limit,90second external timeout.
Only the generated grammar and runner were mounted alongside read-only packages.

All256graphs pass existing strict syntax validation. Each was executed once on
the same handcrafted[[1,2],[3,4]] input in the existing isolated Hodel runtime,
under a240second total admission cap and its per-candidate timeout. All finished
in221.84seconds:1validgrid (identity),255failures,1distinctgridoutput. Failures
retain their slots. This includes non-grid returns rejected by the worker and
possible runtime errors; the wrapper's exit-code records do not distinguish them.
It is NOT a256-task semantic accuracy estimate, held-out result, or proof that
symbolic synthesis cannot work. The order includes all189terminal values before
more expressions; those terminals mostly are not grids.

A separate handcrafted computed-call tree lowered compose(identity,identity) to
x0 and invoked x0 onI. Its exact input grid was returned with verified sandbox
isolation. This extends the prior bridge beyond passing callables as arguments.
All9focused tests pass in1.58s; no containers remain running.

## Decision

Do not send this generic uniform prefix into BED. It has essentially no useful
predictive support at this budget. Next implement source-independent proposal
guidance over the full grammar, with a bounded multi-candidate collector and
explicit canonical-graph deduplication. Compare guided versus unguided search
at equal attempted work on public engineering examples before fresh paid
predictive-transfer qualification. A useful guide may prioritize partial
structures and typed holes, but must not silently drop unmentioned operations,
read target-specific reference solvers, or confuse data-conditioned search costs
with an independent Bayesian prior. No old qualification cohort is reopened.

The current permissive grammar is a fallback vocabulary, not a finished solution
to semantic typing or efficient search. A prospective LLM-native result still
requires calibrated predictive transfer, identical simulated/real branch
updates, persistent worlds separate from policy support, and paired equal-work
myopic/random controls before depth claims. This measured bottleneck is not
itself that result.

Both the previous and current goal turns made concrete progress. No API calls
or new cost; authenticated balance23.609293221, conservative London-day
remaining4.0275717. Goal active/unachieved. No cluster or automation changes.
