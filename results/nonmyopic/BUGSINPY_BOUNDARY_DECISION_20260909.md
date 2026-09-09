# Pre-fix execution boundaries: no horizon claim

Follow-up to the frozen three-case eligibility scope. Read only pre-fix source
and direct dependency definitions; AST inspection never imported or executed the
projects. Exact URLs, hashes, function lines, calls and argument use are in
BUGSINPY_BOUNDARY_AUDIT_20260909.json. Fixed code, patches, tests and endpoints
remain unopened. Previous turn was progress; this turn supplies new source evidence.

## Results

TheFuck25: match uses script and stderr substring predicates. Correction uses one
regular-expression substitution on script. Settings is unused by both underlying
functions. The sudo decorator strips/reapplies a prefix and forwards a Command
namedtuple containing script/stdout/stderr. These inspected definitions have no
filesystem observation or command execution. This is not a safety proof for importing
the entire package. The wrapper retains string/list/bool behavior, so an adapter
must not replace every output by a string or silently discard sudo cases.

Cookiecutter3: read_user_choice validates a nonempty list, constructs an indexed
option map and delegates to click.prompt/Choice. Treating this as a pure list
function would drop retry, EOF, formatting, and input conversion behavior. Any
faithful oracle must explicitly specify the input stream and Click dependency;
there is no justification for hand-writing a convenient replacement.

Black9: format_str has a source-string/mode boundary, but its implementation calls
the parser, target-version inference, line generator and line splitting pipeline.
format_file_contents additionally checks equivalence/stability unless fast mode is
enabled. Importing black.py also loads several third-party packages and OS-facing
modules. Static inspection does not establish hermetic execution. Do not silently
switch entrypoints or disable checks to make a run pass.

## Decision

Do not spend on a TheFuck-only planner: this case's public contract and tiny rewrite
provide weak motivation for deep observation-conditioned hypothesis discovery.
This is a prioritization inference, not a measured zero-gap result. Do not replace
it in the frozen sample or invent hidden configuration flags to create a gap.
Cookiecutter remains unresolved rather than declared pure or excluded for failing
a scientific gate.

Black is the remaining candidate worth an execution-feasibility check because
its semantic behavior is compositional and the public string API could support
executable alternative repairs. That complexity is not evidence of LLM necessity,
posterior calibration or a horizon gap. Before fixed-source access, the next
dependency is a bounded, isolated pre-fix formatting smoke with exact dependency
pins and no model-generated code; retain original mode/parse/error semantics and
record resource limits. Failures are environment failures, not policy losses.

If feasible, freeze a small public-input observation pool and disjoint target
inputs before reading the fixed reference. Keep reference implementation, assertions,
revision IDs and repository history outside the policy workspace. Candidate repairs
must predict reference observations, not just the known buggy implementation's
behavior. Require proposal coverage and fresh-observation value against same-history
redraw before independent joint-transition fidelity and any d1/d2/d3 comparison.
No finite two-patch prior may stand in for the required generative LLM belief space.

The three-case source sample remains Black9/Cookiecutter3/TheFuck25, no replacement.
No paid calls or oracle outcomes were obtained; no monotonicity or positive result
is claimed. Two new tests pass in .09s, including non-execution of a top-level
exception and rejection of missing function definitions. Account usage221.306531939,
balance23.693468061, conservative London-day remaining4.11174654 unchanged.
Full research goal active and unachieved.

Additional host preflight: Docker CLI exists but docker info reports no reachable
daemon. No container ran or image was pulled. macOS sandbox-exec is present but
untested; do not silently treat it as equivalent isolation. Black's pinned setup.py
uses lower-bound dependency requirements (click>=6.5, attrs>=18.1.0, appdirs,
toml>=0.9.4), not a reproducible lock. Establish and verify the isolated runtime
before installing or executing this source; the absence of a daemon is not a
scientific failure or a reason to run generated code unsandboxed.
