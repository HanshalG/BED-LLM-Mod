# Shared-budget two-root switch reference

All four already-opened synthetic histories, both roots in fixed order0,1.
One SwitchReference instance per history: shared five seconds and100000 counted
evaluations, no budget reset between roots. Same17-point partition scan, exact
conditional model, real-line domains, tolerance and cache bounds as the preceding
single-root diagnostic. Execute once from pushed code; retain partial roots as
diagnostic only with status incomplete, action null and numerical_check false.

For complete roots, record outer error and maximum encountered inner error.
The numerical check requires their sum <=1e-5 for every root, as an explicit
conservative reported-error check, not a rigorous global guarantee. No full
panel claim unless all four cases complete and pass. A failure does not license
a larger cap, alternate seed/history, source execution or depth-three deployment.
Even a full pass is only h2 synthetic reference mechanics, not the research goal.
