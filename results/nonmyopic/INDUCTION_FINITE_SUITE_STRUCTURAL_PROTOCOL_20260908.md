# Prospective finite-suite structural audit

This is a new source contract, not a repair of the failed public-generator
contract. Population: all 375 canonical FullObs records at Concept Synth commit
c1f71f98623ab6e3513f820a8e52235d5b498694. No task selection by observed labels,
model success, planning gains, or oracle opportunity. No paid calls authorized.

Authorize exactly one structural audit of the release-manifest-pinned compressed
FullObs file, SHA256
55e21dee281be6ab779119dcce609d7d969e5528863c82a38fa4782fca0100ed.
The evaluator process may materialize the private rows, including formulas and
labels, in memory. It must never print/save raw rows, formulas, labels, world
facts, descriptions or exceptions containing those values. No formula execution,
membership responses, predictions, evaluation caches or planning runs are allowed.
This explicitly supersedes the previous no-row-read restriction for this audit
only. The policy never receives the raw record.

Inspect only record identity/task/schema, reference formula text fingerprints,
formula lexical lengths, and world/domain counts. Normalize formula whitespace
for fingerprints; this is NOT semantic equivalence or alpha normalization.
Group identical normalized formulas before a deterministic hash-based tentative
60/20/20 development/validation/confirmation allocation. Output aggregate counts
only, not task IDs or group hashes. The split is diagnostic, not a final sealed
study split: different syntax can encode the same concept and must be checked
before any generalization claim or experiment.

Report all records; fail closed for unexpected count/schema, duplicate IDs,
missing/invalid structural fields, wrong artifact bytes or output overwrite.
No numerical diversity threshold is a scientific success gate. The audit answers
whether the finite suite visibly escapes the ten-template generator and whether
simple exact-text grouping produces usable block sizes. It cannot establish
non-enumerability, semantic diversity, LLM necessity, prediction or horizon gains.

The empirical population and the agent's prior are distinct. A future finite-suite
experiment cannot quietly give an oracle all reference concepts while describing
its opportunity as attainable by an LLM without that information. Conversely,
any public truth library usable by an LLM must be equally available to classical
controls. A full-library oracle is a diagnostic only. Future permission requires
an explicit information-access contract, semantic split check, independent world
distribution, and complete opportunity/compute controls, all frozen before their
responses. This audit itself opens none of those stages.

Test structural rejection, same-formula grouping, absence of private values in
serialized output, label invariance, deterministic output, byte binding and
overwrite refusal before the single source read. Commit protocol and runner
before running it. Preserve failures rather than inspecting offending rows or
relaxing the parser after outcomes.
