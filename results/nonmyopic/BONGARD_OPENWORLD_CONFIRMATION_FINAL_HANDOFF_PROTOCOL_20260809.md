# Bongard OpenWorld Confirmation Final Handoff Protocol

Frozen: 2026-08-09 Europe/London, before the first paid Bongard response and
before any confirmation block or endpoint outcome.

## Purpose

After all four registered Confirmation96 blocks and the exact combined result
exist, one zero-call terminal handoff must produce the complete analysis and
paper package. This prevents a favorable confirmation disposition from being
assembled differently from a null and removes manual selection among already
frozen secondary analyses.

## Preconditions

Before a combined confirmation exists, preflight may inspect only public file
existence and immutable implementation bindings. It must report the first
missing block and must not load confirmation endpoints.

Once all four block results and the combined result exist, the handoff must
independently replay `verify_combined_result(...)` over the exact ordered block
files. It must reject a missing, partial, reordered, or changed result before
creating a downstream artifact.

## Mandatory Order

For both `full_llm_native_confirmation` and `confirmation_null`, execute or
exactly replay, in order:

1. the DINOv2 plus SigLIP2 classical suite;
2. the path-mediation report;
3. the compute-matched control audit;
4. the endpoint-blind classical horizon-opportunity report;
5. the random-strategy control audit;
6. the shared V7 paper wrapper with all five analysis artifacts.

Every component must refer to the same Confirmation96 result and exact four
ordered block hashes. The paper wrapper must independently replay every saved
analysis. Its TeX, metadata, and headline must replay byte-identically.

## Failure And Replay

The first downstream failure is banked once with the exact successfully
completed prefix. A later invocation may only replay that prefix and return the
same terminal failure; it cannot resume, rerun a paid component, or replace the
failure. A complete terminal result likewise must replay exactly and reject any
changed component hash.

The final handoff makes zero model calls, spends zero dollars, authorizes no
paid call or rerun, and cannot alter the registered confirmation claim tier,
headline rule, gate, task, prompt, seed, action, endpoint, or budget. It reports
the already-determined confirmation pass or null without pooling Development64
and Confirmation96.
