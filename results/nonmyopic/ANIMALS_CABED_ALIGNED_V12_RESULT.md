# Animals CA-BED Aligned V12 Result

Date: 2026-07-24

V12 failed closed during the serving smoke before completing the first shared
tree and before any efficacy endpoint.

- Run ID: `animals-cabed-aligned-v12-smoke-20260724T210824Z`
- Requests: 16
- Reasoning tokens: 0
- Cost: $0.00315648
- Error: a branch requested three valid follow-ups but only two remained after
  the frozen history-duplicate and direct-guess filters.

No formal target was used. The failure does not test aligned likelihoods
against realized utility. It shows that requesting exactly the final menu
width is brittle under strict post-generation validity filtering.

One final distinct serving interface may overgenerate by a fixed two candidates
before applying the unchanged 4/3 menu filter. Failure of that smoke closes the
Animals route.
