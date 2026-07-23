# Range-Gated Rock JSON-Prefix Cluster 26B Smoke Result

Status: **passed every preregistered S0 gate; S1 authorized.**

Fresh seed `24185` completed all ten successor-grounded cells with no invalid
responses. Every cell contained four dynamically legal fixed-root plans, and every
south-root plan was exactly:

```text
move-SOUTH, move-SOUTH, check-5
```

The delayed-route requirement therefore passed `10/10` versus the frozen `8/10`
threshold. The JSON-prefix parser was substantively exercised: two final responses
contained trailing prose after a complete valid object, while eight ended at the
object. The parser retained all raw text and applied the unchanged key, length, root,
and dynamic-legality compiler to the decoded object.

All ten first-stage reasoning generations reached bounded finalization, and all ten
separate final generations returned accepted policies. Accounting:

- accepted logical cells: `10`
- invalid responses: `0`
- physical vLLM generations: `20`
- forced-finalization events: `10`
- prompt tokens: `73,182`
- completion tokens: `42,363`
- scoring-time model calls: `0`
- API cost: `$0`

The full `137,863`-byte reasoning/finalization log, raw finals, compiled plans,
histories, mechanics, and usage are retained in
`range_gated_rock_json_prefix_cluster26b_smoke_20260723/`.

Cluster job: `106385`, `msc`, node `oat14`.
