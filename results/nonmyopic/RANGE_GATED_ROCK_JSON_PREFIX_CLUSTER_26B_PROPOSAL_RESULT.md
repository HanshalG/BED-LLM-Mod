# Range-Gated Rock JSON-Prefix Cluster 26B Proposal Result

Status: **producer passed every scientific gate; frozen exact-identity audit failed on
machine-precision control ties; no trajectory launched.**

Fresh seed `24186` completed all 16 distinct strict horizon-three opportunities with
zero invalid responses. Exact verification selected the LLM delayed-route plan in
`16/16` cells, and every selected plan was the exhaustive open-loop h3 optimum
`move-SOUTH, move-SOUTH, check-5`.

| Registered endpoint | Mean | Paired 95% CI | W/T/L |
| --- | ---: | ---: | ---: |
| LLM h3 minus identical-root random | `+0.480875` | `[+0.479935,+0.481823]` | `16/0/0` |
| LLM h3 minus shared-plan d2 | `+0.485883` | `[+0.484944,+0.487119]` | `16/0/0` |
| LLM h3 minus strong d2 root | `+0.479607` | `[+0.479607,+0.479607]` | `16/0/0` |
| Exact h3 opportunity recovery | `1.000000` | `[1.000000,1.000000]` | `16/0/0` |

The producer passed all five endpoint gates, all mechanics, and exact h3 route-root
selection `1.0`.

## Independent replay boundary

The fresh local replay reproduced:

- every serialized LLM plan and machine-fixed root;
- every exact value and all four aggregate comparison objects;
- all strong-d2 control values; and
- every record after removing only tie-equivalent control-plan identities.

It did not reproduce every control identity exactly. On one cell the cluster producer
selected `check-0` as the strong-d2 root while the local platform selected `check-5`;
their scored h3 values were equal within `1e-17`. On another cell, two matched-random
plans likewise exchanged the selected identity at equal value. Because the frozen
audit required exact root and plan identity, `all_record_fields_match` is false and
the overall independent audit gate remains failed. The audit was not relaxed, the
seed was not rerun, and the conditional trajectory confirmation was not launched.

## Serving

- accepted logical cells: `16`
- invalid responses: `0`
- physical vLLM generations: `32`
- forced-finalization events: `16`
- JSON finals with trailing retained text: `2/16`
- prompt tokens: `117,154`
- completion tokens: `67,204`
- scoring-time model calls: `0`
- API cost: `$0`

Artifacts:

- producer: `range_gated_rock_json_prefix_cluster26b_proposal_20260723/`
- replay: `range_gated_rock_json_prefix_cluster26b_proposal_audit_20260723/AUDIT.json`
- cluster job: `106386`, `msc`, node `oat14`
