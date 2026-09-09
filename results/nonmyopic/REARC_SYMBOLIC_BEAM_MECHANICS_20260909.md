# Productive first-order symbolic control, with explicit scope limits

Implemented a deterministic beam search using generic DSL type annotations, not
selected reference programs. It enumerates grid-valued first-order calls with
grid/intermediate arguments and declared integer/tuple constants. Operations are
derived from the library, not task-specific templates. Round-robin enumeration
prevents a single high-arity primitive from taking every per-depth attempt.

Default search limits:6steps,32beam states,5,000attempts per depth. Candidate storage
is periodically pruned by demonstration fit; at most four forecast graphs return.
Fit combines exact-grid error and cell error on the demonstration bounding canvas.
That heuristic is NOT the held-out predictive score, which remains the separately
frozen whole-grid/fixed900cell scoring interface. No held-out target argument exists.

Two tests verify exact recovery of a two-operation mirror/concatenation composition,
deterministic operation, and attempt bounds. Both pass in .09s. These are hand-built
mechanics examples; no RE-ARC task was run. Trusted DSL execution must take place
inside the isolated worker before use on real demonstrations; this module itself
is not a sandbox or a resource-termination mechanism.

This is a productive but LIMITED symbolic baseline. It does not enumerate
higher-order functions, object-valued intermediates or full128step programs.
Training-behavior deduplication and beam pruning can discard programs that would
generalize differently. Winning against it would not establish that classical
full-DSL synthesis is impossible or that an LLM is irreducible. In particular,
the49step source program lies outside this search envelope. Report this scope,
keep the harder tasks, and retain equal-call blind LLM controls as the direct
test of answer-conditioned proposal value. A headline requires stronger baseline
evidence than this mechanics test.

Next bind the symbolic worker/runtime cap and the actual proposal qualification
weights and example split. Do not execute the benchmark or pay for Luna until
the complete comparison is frozen. No old endpoint or threshold changes.

Previous turn was public-interface progress; current turn implements an actual
non-LLM search control and demonstrates productive composition. Cost$0, account
unchanged at balance23.693468061, daily conservative remaining4.11174654.
Goal active/unachieved.
