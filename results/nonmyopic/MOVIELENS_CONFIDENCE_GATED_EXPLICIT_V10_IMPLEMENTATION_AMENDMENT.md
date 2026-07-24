# MovieLens Confidence-Gated Explicit V10 Implementation Amendment

Date: 2026-07-24

The first serving invocation stopped before model construction and made zero API
calls. The wrapper's fresh-user validator referenced V7's mutable
`ALL_SELECTED_USER_IDS`; the wrapper then temporarily replaced that value with
the V10 cohort, causing the cohort to exclude itself.

The correction snapshots the complete V1-V9 exclusion set at module import and
uses that immutable set during wrapper monkeypatching. A local test reproduces
the exact 46-user selection and smoke prefix `(750, 782)`.

No prompt, model, user, history, seed, margin, threshold, request count, cost
cap, or scientific gate changes. Because no response or endpoint existed, the
same serving smoke remains authorized.
