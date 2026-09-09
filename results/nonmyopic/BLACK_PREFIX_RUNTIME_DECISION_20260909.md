# Pre-fix Black isolated smoke passes

The previously unavailable Docker daemon started through the installed Docker
application. No cluster used. Local free disk was123914300KiB before the pull.
Downloaded Python3.8.20-slim-bookworm, then pinned its resolved digest:
`python@sha256:1d52838af602b4b5a831beb13a0e4d073280665ea7be7f69ce2382f29c5a613f`.

Only pre-fix Black026c81b83454f176a9f9253cbfb70be2c159d822 black.py,
blib2to3 runtime Python/grammar files and LICENSE entered the build. Source hashes
are in BLACK_PREFIX_RUNTIME_SMOKE_20260909.json. No setup.py execution, test bank,
fixed implementation, patches, issue solutions or API credentials were used.
Installed pinned binary distributions of click7.1.2, attrs19.3.0, appdirs1.4.4,
toml0.10.2. These chosen compatible versions are not a claim of reproducing the
original BugsInPy dependency environment; artifact hashes were not separately
locked. Final local image ID is
`sha256:017df47b29fcfc1eeb456f5967ed271adacfac91ffe5757e0efe959cc494ff3c`.

## Observed checks

Used format_file_contents with fast=False and default FileMode, retaining the
upstream equivalence/stability checks. Two deliberately ordinary public smoke
inputs, x=1 and f(1,2) with irregular spacing, produced their expected formatting.
Invalid syntax raised upstream InvalidInput. These inputs are runtime checks, not
policy endpoints or a coverage/saturation test.

Runtime used user65534, network=none, read-only root, all capabilities dropped,
no-new-privileges,32processes,256MiB,1CPU,16MiB temporary filesystem, no host mounts.
Direct network and root-write probes failed; OPENROUTER_API_KEY was absent.
Container exited and docker ps showed no running containers. This is a bounded
smoke of reviewed upstream source, not a general proof against arbitrary hostile
candidate programs. Python3.8 is old; use stronger adversarial isolation checks
before permitting model-generated code. The launcher timeout bounds the CLI wait
but does not yet guarantee container cleanup on a timeout; add/test explicit
container ownership and cleanup before any such deployment.

Two local source-selection tests passed in .16s. Runtime result status passed.
Docker application remains running; built image retained for future read-only
work. Temporary build context was removed by its owning temporary-directory scope.

## What this authorizes

Only prospective experiment construction, not a paid policy sweep. Next freeze a
public input-domain/candidate-query/held-out-input split without reading the fixed
formatter. Define reference outcomes (formatted text or exact error category),
public semantic context and candidate repair scope. Reference code must be
evaluator-only; never expose git metadata, fixed assertions or target outcomes.
First establish whether generated repairs gain predictive coverage from purchased
observations over same-history redraw and productive executable controls. No depth
experiment until that and independent transition fidelity clear their gates.

Do not turn the two passing smoke examples into a small positive benchmark. Neither
an LLM role, non-myopic opportunity, nor monotonic depth improvement has been
demonstrated. The frozen three-case source sample is unchanged; Black is receiving
feasibility work, not outcome-selected replacement of the other cases.

Previous turn was progress; current turn removes a verified runtime obstacle.
Modelcalls0/cost0. Authenticated usage221.306531939/balance23.693468061;
London Sept9 conservative remaining4.11174654 including prior .04 uncertainty.
Full goal remains active/unachieved.
