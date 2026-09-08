# Typed continuation interface: zero-call source-language verification

The preceding turn made scientific progress by measuring a specific interface
failure. The original proposal gate remains terminally closed. This implementation
is a new representation, not a repair, reinterpretation or rerun of its responses.
No model call or new scientific outcome was opened in this work.

## Change

Represent each program as nested objects containing a canonical `statement` and
`next` continuation. Each reachable type state enumerates exactly the source's
type-valid operation/lambda/argument tuples that consume the previous result.
The schema permits termination only after 2, 3 or 4 statements. Later statements
can still use ALL earlier variables as additional arguments: this does not narrow
the original grammar to unary chains. Final output may be int, list or ERROR.

The generated JSON Schema has 31 reachable type states, 587 statement enum
entries and 19,083 compact-JSON bytes. Canonical sorted compact schema SHA256:
1bbd8edf7ca1c0dad7c00dd8bea676b03732dfd29627c8feac61fe34a82d3332.

Local decoding checks the same transitions independently of provider behavior.
It never inserts missing arguments, changes operations, rewrites old JSON, repairs
wrong references or salvages valid members of an invalid batch. Duplicate source
syntax is returned once, matching the existing proposal interface's pool rule.
Programs satisfying the schema are source-valid by construction. This does NOT
claim that an arbitrary remote provider will support or enforce this schema.

Every source program has one canonical continuation path. Encoding does not
change its source-prior probability or executable semantics. The LLM's sampling
distribution would still not be the source prior, and a finite generated pool
would still not be a calibrated full-grammar posterior.

## Evidence

33 focused constrained/prior/proposal tests passed in1.58 seconds; lint passed.
Checks include all reachable type-state choice sets against the source grammar,
100 source-sampler structural paths with identical round-trip syntax and exact
prior probabilities, independent JSON Schema validation, early/late termination,
scope/type/previous-result errors, duplicate syntax and whole-batch rejection.
Only structural or constructed fixtures were used; no held-out target labels.

The complete schema is larger than the prior gate's 16KB message-content limit.
A successor must explicitly budget schema plus message/framing input and verify
actual provider schema support. Do not silently insert this into the old runner,
reinterpret the old failed response or inherit its consumed spending approval.

## Remaining dependency

This removes a mechanically enforceable representation failure; it does not
demonstrate evidence-responsive proposals, held-out predictive benefit, calibrated
beliefs, a numerical planning reference or a non-myopic result. A prospectively
new successor must retain those scientific tests and controls. Grammar compliance
alone should not become the claimed semantic accomplishment.

No additional paid scope is authorized. Automation remains paused and the full
goal remains incomplete. London time at the authenticated check was still Sept8
(19:01 BST): credits245, usage220.376763864, balance24.623236136. The previous
$0.00006987 accepted cost has now posted and matches the frozen daily ledger.
