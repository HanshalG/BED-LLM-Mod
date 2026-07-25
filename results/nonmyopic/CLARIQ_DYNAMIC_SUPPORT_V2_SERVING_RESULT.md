# ClariQ Dynamic-Support V2 Serving Result

## Decision

The fresh one-call V2 serving gate passes every frozen condition. This
authorizes a separately implemented and preregistered full mechanics tree on
topic `60`. The serving response will be discarded rather than reused.

## Results

| Metric | Result | Gate |
|---|---:|---:|
| Physical requests / HTTP attempts | `1 / 1` | exact |
| Parsed support lines | `8 / 8` | exact |
| Positive response profiles | `8` | at least `4` |
| Support entropy | `1.843982` nats | at least `1.0` |
| Myopic EIG range | `.618601` nats | at least `.05` |
| Branch requests | `0` | exact |
| Endpoint loads | `0` | exact |
| Retries / reasoning / forced exits | `0 / 0 / 0` | exact |
| Cost | `$0.009445` | at most `$0.05` |

The result establishes that the single-space code protocol is exact and that
the initial open-intent support is neither probability- nor response-profile
collapsed. It is not a first-link efficacy result.

## Artifacts and Budget

- Public artifact:
  `results/nonmyopic/clariq_dynamic_support_v2_serving/clariq-dynamic-support-v2-serving-20260725T235057Z/SERVING.json`
- Public SHA-256:
  `65efd9fe56d66a9f8547f0978ffabd05d770965a8906e83f20b2d4724f7b7eae`
- Private raw SHA-256:
  `9683ba680d1a196b8bfc1a476ec2d4374d3c0388b76f0784548117bb6b695bdb`
- Prompt / completion tokens: `2,320 / 243`
- Project-ledger spend / headroom:
  `$91.91762070922331 / $13.082379290776686`

Development topic `148` and holdouts `102/11/103/141` remain untouched. The
`$25` Monday reserve is intact. OatML use: none.
