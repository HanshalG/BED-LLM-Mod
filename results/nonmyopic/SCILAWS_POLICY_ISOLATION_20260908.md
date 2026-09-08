# Local policy isolation verified

Previous turn: concrete measurement-accounting progress. This turn adds
`environments/scilaws/policy_isolation.py`, a macOS subprocess boundary with
no unsandboxed fallback. It is not a benchmark launcher or paid authorization.

The policy receives an explicitly prepared public directory and bounded stdin.
Its environment is reconstructed rather than inherited; OpenRouter credentials
are not forwarded. File contents are readable only within the public directory,
installed system/Homebrew runtime paths, and specified device files. The profile
does not allow all of `/System`, which would include the writable Data volume.
Network, file writes (except null output), and process forking are denied.
File metadata access is allowed; this is not metadata confidentiality.

Wall-clock and CPU limits terminate execution. Regular inherited output files
have OS-enforced size limits; oversized output fails rather than being parsed
as a valid policy response. Core dumps are disabled. Errors returned to the
caller are generic rather than reflecting raw subprocess output. Missing
platform support fails closed. No container runtime was started: the installed
Docker client could not reach a daemon, whereas native sandbox execution worked.

## Verification

48 combined SciLaws tests pass in2.39s; lint passes. Isolation tests run on this
host, not mocks: public JSON input/read success, credential omission, private
file denial, symlink denial, Data-volume-alias denial when available, network
denial, read-only public files, fork denial, timeout and output-limit failures.
Unavailable-backend behavior is tested separately by platform substitution.
No benchmark secret or real credential is used in adversarial probes.

Initial profiles prevented even runtime startup. Allowing the root directory
itself, not the user's filesystem tree, fixed startup. The test interpreter is
the resolved Homebrew base Python, not a user-cache virtual environment requiring
broader home-directory access. These were pre-experiment software corrections.

## Scope and next dependency

The caller must stage only verified public files, without hidden-data hard links,
under the allowed directory. Installed runtimes are trusted. This is a tested
access boundary, not a formal security proof or a full memory-resource sandbox.
Its timeout applies to the policy subprocess, not the evaluator's simulator.
The measurement object's backend timeout and an end-to-end broker still need
integration before benchmark execution. Other operating systems require their
own tested backend, not an automatically weakened profile.

Scientific blockers are unchanged: per-task licensing/support verification,
public-only agent prior and observation model, fixed target measure, bounded
complete opportunity panel, then a fresh proposer/predictive calibration gate.
No published task outcomes or simulator states were loaded, no model calls were
made, and no old failed route was reopened. Authenticated account/ledger remain
usage220.376693994, balance24.623306006, zero spend. Automation remains paused;
the full non-myopic research plan remains unfinished.
