# HoVer Semantic-BED Manifest Result

## Outcome

**Pass.** The official HoVer development source reproduces the preregistered
content-blind 3/4-hop partition exactly. No claim, label, or supporting fact was
emitted by the manifest.

## Integrity

- Official repository revision:
  `39b84697f196308f398a251a7aea9b82ae0f0562`.
- Source SHA-256:
  `67c14858f2d7fcdb96b6fe3d538ffcd6f76e3ba594aa2c0cd4359f601101e89d`.
- Source rows: 4,000.
- Eligible rows: 1,835 three-hop and 1,039 four-hop.
- Manifest SHA-256:
  `d64db625ca142b97df990b1831b62c1d4438b1f52555c09388e2f1762f733b13`.
- `content_emitted`: `false`.

| Split | Three-hop | Four-hop | Total | Ordered UID hash |
| --- | ---: | ---: | ---: | --- |
| Mechanics | 3 | 3 | 6 | `5ff4651736dd93dd91d793fa084ded2afd9f7b7a739020094a42be114aa17f4c` |
| Opportunity | 200 | 200 | 400 | `230ba25090172484250812ae0c24b089b2f3c888c7406702a47d2cd18a570fd0` |
| Development | 20 | 20 | 40 | `2eac3af917f9647919286ef37c1f691f0d95248d3e26e9cd353900ec3d0351f9` |
| Holdout | 1,612 | 816 | 2,428 | `050dc22e6700618f8929d301cfb7528f2b95bf77759077584d433064770cac6f` |

## Mechanics Inspection

Only after the manifest reproduced, the six mechanics rows were opened. The
supporting-fact annotations resolve to exact document titles and sentence
indices. Article observations resolve against the official 2017 HotpotQA
Wikipedia database:

- database bytes: 2,156,273,664;
- database SHA-256:
  `c37ee397916ec0bffacfe8902db454a5cda88a7a188409217b2e15231fe5ee2f`;
- SQLite integrity check: `ok`;
- documents: 5,233,329.

The official dev TF-IDF artifact contains 100 ranked document titles and scores
per task, but no article text. Its SHA-256 is
`b50a961f63a95ff184986af766b15ff6b1d6c98f7e86b39355b64b1a85fb3745`.

An exploratory mechanics-only transition used the top ten claim-retrieved
documents as roots and exact title mentions among the frozen top 100 as legal
continuations. One of six rows showed a strict depth-3 root reversal after
giving the myopic root its own best possible continuation tail. This is enough
to define a prospective prevalence audit, not evidence for a policy result.

## Authorization

The manifest pass authorizes a separately committed, zero-call structural
audit on exactly the 400 opportunity rows. Development and holdout values
remain sealed. No OpenRouter call or OatML job is authorized.
