# RegretBench SMC Support-Recovery Contingency

Date: 2026-08-07

Status: prospectively frozen before any RegretBench support or policy response.

## Motivation

The primary RegretBench support instrument regenerates all eight semantic
hypotheses after an answer. This measures path-dependent belief generation
directly, but it can discard a previously useful hypothesis through sampling
noise. LLM-SMC-S instead keeps a finite particle population and asks the LLM
to revise low-quality particles while retaining useful ones. The contingency
tests whether explicit parent-particle continuity preserves existing truth
coverage while still allowing an answer to recover missing interpretations.

This is an independent fallback, not a repair or alternate analysis of the
primary Aug 8 result. The primary frozen chain and its interpretation remain
unchanged.

Primary references:

- BED-LLM: <https://arxiv.org/abs/2508.21184>
- Doing Experiments and Revising Rules:
  <https://proceedings.neurips.cc/paper_files/paper/2024/file/5f1b350fc0c2affd56f465faa36be343-Paper-Conference.pdf>

## Authorization Boundary

The contingency is sealed unless all of the following are true:

1. The Aug 8 primary support-recovery development result exists and has a
   literal `gated_null` status.
2. Its mechanics gates all pass and its independent `VERIFICATION.json` is
   `verified` with no replay mismatches.
3. The primary dynamic policy and confirmation were therefore never opened.
4. The contingency runs no earlier than 2026-08-09 Europe/London, from a new
   account-wide daily ledger and within the same hard `$5.00` daily cap.

A primary mechanics failure, partial artifact, missing verification, primary
support pass, or any opened policy artifact authorizes nothing. The
contingency never runs on Aug 8 and cannot replace, rescue, pool with, or
reclassify the primary support result.

## Frozen Instrument

- Source cohort: the same ordered 64-task RegretBench development split.
- Model: `deepseek/deepseek-v4-flash-0731`, reasoning disabled.
- Parent particles: reuse the exact eight raw initial particle slots already
  banked for each of the 64 tasks by the verified primary support run. Repeated
  raw slots remain distinct particles, as in a resampled SMC population; the
  separately parsed/deduplicated support remains the root-coverage endpoint.
  No initial-support calls are repeated.
- Root query and official reply: reuse the exact banked private controls from
  the primary run. No truth is exposed to the model.
- Arms: one answer-conditioned child population and one history-blind child
  population per task, adjacent and using the same requested seed.
- Seeds: a new disjoint frozen range beginning at `202608270000`.
- Calls: exactly `128` model requests, concurrency at most `24`.
- Temperature: `0.7`; maximum output: `2200` tokens.
- Run cap: `$0.50`; projected cost: `$0.30`.

Each child response must contain exactly eight unique hypotheses and exactly
four distinct single-dimension questions. It receives the complete parent
population with public model-generated interpretation, answer, and weight.
Every child has a unique `parent_index`; the eight indexes must be the exact
permutation `0..7`.

Each child declares `revision_type`:

- `retained`: normalized interpretation and final answer exactly equal its
  parent;
- `revised`: normalized interpretation or final answer differs from its
  parent in light of the visible history.

Every response must retain between two and six parents inclusive, so it both
preserves continuity and performs genuine revision. Child weights are
renormalized after strict parsing. Duplicate child interpretation/answer pairs,
invalid lineage, false retention labels, unchanged revisions, malformed
questions, or extra fields fail mechanics rather than being repaired.

## Frozen Endpoints And Gates

Truth coverage uses the same conservative lexical alias matcher as the primary
support gate. Only official supported root queries enter science. The public
result records child lineage diagnostics, coverage indicators, and hashes, but
not hidden aliases, truth identities, root question text, or parent content.

Mechanics requires:

- exact `128` accepted requests and HTTP attempts;
- zero retries, provider-error retries, reasoning tokens, and forced exits;
- all responses strict, eight-unique, exact-parent-permutation outputs;
- two through six exact retained children per response;
- conditioned/blind adjacency and identical paired seeds;
- at least 48 supported roots; and
- all payload privacy and parent-provenance audits passing.

Science requires the conjunction of:

- at least 48 supported tasks and at least 16 supported root-missing tasks;
- conditioned-minus-blind coverage at least `+0.05` overall;
- 20,000-paired-bootstrap probability of positive overall gain at least
  `0.80`;
- conditioned recoveries strictly exceeding conditioned losses;
- conditioned-minus-blind coverage at least `+0.10` on root-missing tasks;
- conditioned coverage at least `0.90` on root-covered tasks; and
- conditioned root-covered losses no greater than history-blind losses.

A literal pass authorizes only a separately preregistered SMC-style dynamic
policy experiment. It does not authorize the existing policy, confirmation,
or any paper claim. Null and failure are banked once without prompt tuning,
threshold tuning, reruns, or favorable-subset rescue.

## Verification And Cost Boundary

An independent zero-call verifier must reload the primary raw initial supports
and controls, hash-bind every parent population, reparse all 128 raw child
responses, reconstruct lineages, coverage, bootstrap statistics, gates, and
status, and hash-bind the reported result. The daily executor must reserve the
full `$0.50` before dispatch and reconcile account-wide spend as the maximum of
posted and local accepted-request cost.

Freezing and implementing this contingency makes zero model calls and opens no
primary, contingency, policy, confirmation, or endpoint artifact.
