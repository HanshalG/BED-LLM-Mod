# ClariQ Dynamic-Support V2 Manifest Amendment

## Reason

V1 failed before any branch or endpoint because GPT-5.4 separated an otherwise
complete response-code sequence with single spaces. The eight lines, masses,
hypotheses, and all 13 codes per line were present. No V1 response is cleaned,
reused, or scored, and topic `38` is not rerun.

One final transport V2 is allowed on a fresh, already-disclosed mechanics topic.
The sole interface change is:

```text
V1: ABCCBAC...
V2: A B C C B A C ...
```

The V2 parser requires exactly one ASCII space between codes and still rejects
missing, extra, or invalid codes. This is a new response, not repair of V1.

## Task Boundary

- V2 mechanics topic: `60`, selected from the already-open opportunity split.
- Topic `60` has 15 valid roots, 90 root-answer branches, and 91 calls in a
  complete dynamic-support tree.
- Topic `38` is closed and not rerun.
- Development topic `148` is referenced only by ID; its content is not reopened
  and its utility remains unread.
- Holdout topics `102`, `11`, `103`, and `141` remain content- and
  utility-sealed.

The parent manifest SHA is
`8871d72aca825aa5b34cf52295eb93fe79caf631c0bba070d82abf6c3bec698d`.

## Gate

Manifest generation must reproduce all source hashes, split memberships, parent
manifest bindings, and exact V2 structural counts without API calls or utility
value access. Passing authorizes only a separately frozen one-call target-free
serving gate. It does not authorize the 90 branch calls or an NDCG endpoint.

## Budget

Manifest generation costs `$0`. The stricter project ledger has
`$13.091824290776685` remaining; the `$25` OpenRouter reserve through Monday
2026-07-27 remains protected. OpenRouter is the only possible later compute
path, and OatML/cluster use is prohibited.
