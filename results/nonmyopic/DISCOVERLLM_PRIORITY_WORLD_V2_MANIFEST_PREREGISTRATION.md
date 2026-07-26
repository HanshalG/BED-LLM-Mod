# DiscoverLLM Priority-World V2 Manifest Preregistration

Frozen before downloading or inspecting the Technical Writing and SVG Drawing
Parquet shards.

## Question

Does the full released DiscoverLLM source contain enough target-blind,
path-dependent semantic state to support a new hidden-priority-world BED
construction without using released preference scores as policy evidence?

This is not a replay of DiscoverLLM and not a repair of the failed
Creative-only V1 threshold. V2 broadens the source to all three official
domains and uses each artifact's earliest released two-candidate turn, which
may occur after turn one because the public dataset filtered some earlier
turns.

## Frozen Eligibility

1. Pin code `a9eb2846` and dataset revision `c857bbf6`.
2. Use Creative Writing, Technical Writing, and SVG Drawing.
3. Exclude the four Creative Writing artifacts inspected during source audit:
   `artifact_1`, `artifact_11`, `artifact_151`, and `artifact_159`.
4. For each domain/artifact, select its earliest released turn only.
5. Require exactly two distinct candidate completions with an identical
   pre-action `criteria_history`.
6. From the current pre-action state, retain hierarchy roots with
   `aware < 1`, at least three nodes, and depth at least two.
7. Require at least four eligible roots. Select exactly four by the fixed seed
   `24412` using only domain, artifact ID, and root ID.
8. Treat those four roots as a uniform prior over mutually exclusive hidden
   user-priority worlds.
9. Do not load released candidate scores or winner labels.

## Content-Sealed Split

Shuffle eligible `domain:artifact_id` keys with seed `24412`:

- mechanics: 3;
- opportunity: 60;
- development: 30;
- holdout: all remaining.

The public manifest may emit domain-prefixed artifact IDs, structural counts,
and ordered/world-selection hashes. It must not emit prompts, hierarchy text,
candidate completions, source metadata, selected root IDs, released scores, or
winner labels.

## Structural Gates

All are conjunctive:

1. all three source hashes match the pinned release;
2. at least 250 artifacts are eligible;
3. every split is nonempty;
4. every mechanics artifact has at least one candidate completion containing a
   question;
5. all selected worlds satisfy the frozen hidden/depth/size criteria;
6. exactly two shared root actions and four latent worlds exist per artifact;
7. OpenRouter calls and OatML jobs remain zero.

A pass authorizes only a separately implemented and preregistered three-task
mechanics smoke. It does not authorize opportunity, development, or holdout
endpoints. A failure closes this exact V2 construction with no threshold,
domain, turn, or world-count repair.
