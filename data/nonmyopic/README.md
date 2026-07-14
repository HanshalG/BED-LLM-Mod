# Frozen UCI Zoo Matrix

`uci_zoo.data` is the 101-row `zoo.data` file from the UCI Machine Learning
Repository Zoo dataset, downloaded on 2026-07-14 from
`https://archive.ics.uci.edu/static/public/111/zoo.zip`.

- Source citation: Forsyth, R. (1990). Zoo [Dataset]. UCI Machine Learning
  Repository. https://doi.org/10.24432/C5R59V
- License: CC BY 4.0.
- SHA-256: `cddc71c26ab9bc82795b8f4ff114cade41885d92720c6af29ffb69bcf73f0315`

The oracle control treats animal identity as the hidden target and turns the 15
Boolean attributes plus six exact-leg-count predicates into a deterministic
Yes/No question matrix. It does not query an LLM. The matrix is vendored and
hash-checked so every trial has a reproducible scripted answerer.
