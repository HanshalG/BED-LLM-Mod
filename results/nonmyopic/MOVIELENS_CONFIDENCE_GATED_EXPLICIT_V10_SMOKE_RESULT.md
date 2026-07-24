# MovieLens Confidence-Gated Explicit V10 Smoke Result

Date: 2026-07-24

Run: `movielens-confidence-v10-smoke-20260724T224322Z`

Status: **passed; the frozen 44-user screen and conditional 16-user formal
confirmation are authorized.**

The fresh-history interface completed exactly 12 requests with zero reasoning,
structured retries, forced exits, or runtime failures. It generated six initial
profiles, all five hypothetical rating transitions for the one smoke
candidate, and a finite explicit rollout score.

The smoke cost `$0.05382851`. Linear request projection for the 856-call formal
run is `$3.83977`, below the frozen `$7.00` cap and current project headroom.
Private raw responses are checkpointed under SHA-256
`490acfbf4dab8842bbef55ae03c5ad659e47a3df62100c901c994cf62353b9cb`.

The smoke contains only one candidate and therefore no confidence-selector
efficacy endpoint.
