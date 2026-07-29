# AllenAI TwentyQuestions Source Opportunity Screen

Date: 2026-07-29

Status: **reproducible semantic-likelihood source, but no non-myopic planning
opportunity; route closed**.

## Source

- repository: `https://github.com/allenai/twentyquestions`
- audited commit: `7aee4ec00ba2dbd376bdb626c4d4edd3c7e246be`
- license: Apache-2.0
- official data archive SHA-256:
  `5e670110357f45064b82183896385bba0c9cddae3c404125387d864ad5dcb78f`
- `twentyquestions-all.jsonl` SHA-256:
  `059a91cad9ae6d09084956c5e45c7eecd2c9dbe3f8748dfd1555d7c4d1c0f187`

The release contains 78,890 high-quality human-labeled rows over 8,806
subjects and 24,924 literal question strings. It includes both answers from
real games and additional counterfactual subject-question labels collected to
reduce subject-only bias.

## Counterfactual Matrix Audit

Questions were canonicalized by lowercasing, extracting alphanumeric tokens,
and joining them with single spaces. This gives:

- 19,995 canonical questions;
- 76,078 canonical subject-question pairs; and
- 1,118 canonical questions observed for at least eight subjects.

The audit exhaustively closed the 1,118 subject-support sets under
intersection, retaining intersections with at least eight subjects. There are
7,380 distinct support intersections. The largest complete eight-or-more
subject submatrix has only six question columns. Those columns are mostly
near-duplicates such as `is it alive`, `alive`, `is it human`, and `is it a
person`.

Across all complete support intersections, only one has at least eight
distinct human-answer signatures:

- subjects selected deterministically by signature:
  `lava`, `candy`, `woodpile`, `loot`, `knife`, `box`, `chair`, `parent`;
- questions:
  `is it alive`, `is it an object`, `is it made of metal`,
  `is it made of wood`, `is it made of plastic`.

Exact dynamic programming with a uniform prior and deterministic majority
answers identifies the same root action, `is it made of metal`, for greedy,
depth-2, and depth-3 entropy minimization. Remaining expected entropy is
`1.386294`, `0.693147`, and `0.0` nats respectively, but deeper planning does
not change the first decision.

## Consequence

The source could test an LLM's semantic answer likelihoods, but it does not
contain enough shared human counterfactuals to test whether non-myopic
planning changes or improves the root decision. Filling the sparse matrix with
an LLM would make both the simulator and the realized endpoint synthetic and
would discard the main value of the human release.

No OpenRouter calls, serving smoke, matrix completion, question paraphrase
pooling, or paid policy experiment is authorized for this route.

