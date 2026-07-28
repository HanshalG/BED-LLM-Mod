# Number Game Pooled-Support Development Result

Date: 2026-07-28

This is a post-hoc, zero-call mechanism audit on the two already-open 32-tree
datasets. It motivates but is not pooled into the fresh confirmation.

## Control

For each tree, deduplicate every hypothesis generated under every candidate
root and answer into one global static support. For each contemplated query,
filter that same support by the observed label and perform exact depth-two
terminal-risk scoring. The control therefore receives 134--221 unique
LLM-generated rules, but cannot use root-conditioned generation.

Pooling only the two answers within each root is an exact identity on all
64 trees: answer-consistency filtering reconstructs the original branch
supports. The load-bearing transition is root-conditioned generation, not
bookkeeping of which answer produced a logically consistent rule.

## Results

| Open source | Global pool size, mean | Roots differ | Candidate Brier | Global-pool Brier | Gain | Whole-tree 95% difference CI |
|---|---:|---:|---:|---:|---:|---:|
| Gemini planner / GPT targets | 150.2 | 25/32 | 0.18359 | 0.20777 | 11.64% | [-0.03528, -0.01357] |
| GPT-mini planner / Gemini targets | 192.2 | 27/32 | 0.18898 | 0.21594 | 12.49% | [-0.04483, -0.01274] |

Hamming reductions are `17.91%` and `22.81%`, with wholly negative
whole-tree intervals. Coverage and extension-novel Brier/Hamming differences
favor root-conditioned generation in both sources.

## Interpretation

The existing result is not explained by giving the candidate access to more
hypotheses than a static planner. A much larger global pool performs worse.
The post-hoc status means this is mechanism development, not confirmation.

Result SHA-256:
`182c900cbddb4ba3b00584662a8bbe90ec154761f3ae60cb04293d1088abeab4`
