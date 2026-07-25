# BrowseComp-Plus Path-Opportunity Result

## Decision

The zero-cost opportunity audit passes every preregistered gate by a wide
margin. BrowseComp-Plus has abundant observation-enabled semantic evidence
paths and authorizes a separately frozen mechanics smoke on the five already
open tasks.

## Results

- Exact opportunity tasks: `120`.
- Usable completed trajectories: `120`.
- Tasks retrieving any official evidence: `89`.
- Tasks with an observation-enabled evidence bridge: `75`.
- Tasks with an observation-enabled gold bridge: `65`.
- Tasks first reaching evidence on search 3 or later: `39`.
- Tasks whose bridge follows an immediate no-new-evidence search: `59`.
- Bridge tasks by low/mid/high evidence-count bin: `20 / 28 / 27`.
- OpenRouter calls / OatML use: `0 / none`.
- Public analysis SHA-256:
  `b7fe444d1a9b7b1f312f54a4c75ca141d703ca23c35c3558156ecb3d91fb5063`.

## Interpretation

This is structural evidence, not a method result. A bridge query reuses at
least two semantic tokens absent from the question but available in earlier
returned snippets, then retrieves unseen official evidence. The audit proves
that useful continuation actions are available from observations across the
benchmark; it does not prove the released GPT-5 agent causally relied on
those terms or that our planner can rank the paths.

The next gate must therefore test the first link directly: whether an
LLM-generated non-myopic strategy score ranks complete two-step paths by
realized evidence value better than its own direct/root score.
