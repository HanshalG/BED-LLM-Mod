# Weighted bounds: tails controlled, central uncertainty remains

Frozen implementation/protocol caf54883. Artifact SHA256
038d2711d923c2c26005f7d0f5ced4285351fed4f819866bd3fdcfcc999a22e7:
SCILAWS_PARTICLE_WEIGHTED_BOUNDS_AUDIT_20260908.json.
All48 initial fixtures,32 children and8 action intervals per child completed.
All1152 saved independent continuation values lie within their analytic intervals
(allowing saved numerical reference errors). No independent integrals or correction
plans were repeated. Process exited normally.

| Diagnostic | Result |
| --- | --- |
| Full terminal width range | 0.0048570-0.0847567 |
| Full terminal budget met without refinement | 0/48 |
| Extreme-tail contribution to interval width | 1.9580e-7 to3.7154e-6 |
| Ideal exact child-minimum refinements per fixture | 15-20 |
| Total ideal exact child-minimum refinements | 826 |

Extreme tails here are all8 nodes belonging to fixed probability intervals
[0,1e-5] and[1-1e-5,1], not just the failed nodes. Their full width contribution
is below3.72e-6 in every fixture, so half-width is below1.86e-6. Under this fixed
outer numerical rule, conservative tail intervals could consume only a small
part of the5e-5 terminal-error budget. This is a derived interval result, not
an assumption that rare observations can be ignored.

The broad central intervals, however, prevent every complete root enclosure
from meeting that budget. The idealized15-20 refinements assume a whole child
minimum becomes exact for free; each may actually require resolving multiple
actions. These are lower-work diagnostics, not attainable runtime estimates.
Only predecessor action0 was considered. Outer integration error is unbounded,
so none of these figures qualifies a complete h2 decision.

## Decision

Do not spend more effort making all extreme child approximations uniformly
precise. Instead investigate whether central child minima can be certified
cheaply, using tighter lower bounds or explicitly charged adaptive refinement.
A working policy would need full-root coverage, complete outer error accounting,
and the original runtime/storage limits. This diagnostic supplies no permission
to execute the failed fixed-rule h2 path or relax the previous local screen.
It is not evidence that deeper planning improves predictions.

The larger scientific task remains untouched: source-law calibration, useful
LLM-generated models and paired non-myopic policy evidence are still absent.
Five focused tests passed0.70s; scoped lint passed. Bounds use ordinary floating
point plus padding, not rigorous directed rounding. Source/model calls0, cost0;
authenticated account245/220.376693994/24.623306006 and London Sept8 ledger
unchanged. Automation remains paused and the full goal remains unfinished.
