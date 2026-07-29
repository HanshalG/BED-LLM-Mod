# Number Game Classical-Grammar Irreducibility Audit

Status: frozen before constructing the grammar bank or measuring any artifact
overlap.

Date: 2026-07-29

Cost: zero model calls and zero OpenRouter spend.

## Question

The current Number Game result shows that an LLM-generated, path-dependent
depth-three belief tree outperforms an equally cross-fitted depth-two selector
on independent target concepts. This audit asks a narrower reviewer-facing
question:

> Does that depth effect remain on semantic concept extensions outside a broad,
> deterministic classical grammar bank fixed without inspecting the LLM
> expressions?

This is an extension-level audit. Two syntactically different expressions with
the same truth set on the domain are the same hypothesis.

## Frozen Evidence

The fixed policies and Qwen endpoint targets are not regenerated or reselected.
The audit is bound to:

- Qwen cross-judge result:
  `a7e0549f2f9ff7b1c2394ebe076bdbf1b70799479a3597e01bf66b7003edc1cb`
- Qwen cross-judge endpoints:
  `647de3c6561ff917691dc3c14176dc4007f90b17230bbb6ee690468d978a613d`
- fixed-policy source-one trees:
  `cf239683033be3fbfed6a449f2aaa1ad1bcbe57d935ca21cf43e46ba938ea7f9`
- fixed-policy source-one result:
  `47e684b11ac5c6340ff4f063ff7a14db8b8d0bd18b184d61f91e315903e4809e`
- fresh-replication source-two trees:
  `197dcfe3cb48eeb9a4b656d0f9b20ef2e0b45ac8d1df0ec4bd9358e07af18802`
- fresh-replication source-two result:
  `25e0939164f6806b30481e48a88372f22639f6065096cb59978600509d43a3d8`

There are 64 fixed trees in two independent 32-tree source studies and 16
Qwen endpoint-support draws per tree. Existing selected roots are used exactly.

## Frozen Classical Grammar

All rules are evaluated on the integer domain `0..100` and stored as 101-bit
extensions. Empty and universal extensions are removed. Duplicates are removed
by extension.

The core atom set is:

1. `n < c` and `n <= c` for every integer `c` in `0..101`.
2. `n mod k == r` for `k=2..20` and `r=0..k-1`.
3. `digit_sum(n) < c` and `digit_sum(n) == c` for `c=0..20`.
4. `digit_sum(n) mod k == r` for `k=2..9` and `r=0..k-1`.
5. `n` ends in decimal digit `d` for `d=0..9`.
6. `is_square(n+s)`, `is_prime(n+s)`, and `is_power_of_two(n+s)` for
   `s=-12..12`.

For every unordered pair of distinct deduplicated core atoms `A,B`, add:

- `A and B`
- `A or B`
- `A and not B`
- `B and not A`
- `A xor B`
- `A xnor B`

The extended single-atom set is:

1. `(a*n+b) mod k == r` for `a=1..5`, `b=-10..10`, `k=2..15`, and
   `r=0..k-1`.
2. `is_square(a*n+b)`, `is_prime(a*n+b)`, and
   `is_power_of_two(a*n+b)` for `a=1..12` and `b=-50..50`.
3. `(n-b)` is divisible by `a` and
   `is_square((n-b)/a)`, `is_prime((n-b)/a)`, or
   `is_power_of_two((n-b)/a)` for `a=1..12` and `b=-30..30`.

Finally, add the complement of every core atom, pairwise expression, and
extended atom. No constants, parameters, expression fragments, or extensions
will be added after artifact overlap is observed.

The bank is canonicalized by sorting its integer bit masks. Its public SHA-256
is computed over a version tag followed by each mask encoded as 13
little-endian bytes. The bank mechanics gate requires at least 100,000 unique
nonconstant extensions.

## Frozen Analyses

### LLM support novelty

For each source tree, classify the initial support, generated first-refresh
supports, and generated second-refresh supports by exact extension membership
in the classical bank. Report both occurrence-weighted and globally unique
counts.

The support-novelty gate requires:

- at least 5% of globally unique generated second-refresh extensions are absent
  from the bank;
- at least 48 of 64 trees contain at least one grammar-novel generated
  second-refresh extension; and
- each 32-tree source study contains at least 20 such trees.

### Independent endpoint novelty and power

Classify every Qwen endpoint hypothesis by extension. Within a tree, a draw is
nonempty when it contains at least one grammar-novel target.

The endpoint-power gate requires:

- at least 5% of endpoint hypothesis occurrences are grammar-novel;
- at least 512 grammar-novel target occurrences overall;
- at least 48 of 64 trees have at least 8 of 16 nonempty draws; and
- each source study contributes at least 24 analyzable trees.

Nearest-bank Hamming distances for grammar-novel unique endpoint extensions are
descriptive and cannot rescue a failed gate.

### Fixed-policy efficacy on grammar-novel targets

For each nonempty draw, evaluate the already selected cross-fitted depth-three
and cross-fitted depth-two roots on only grammar-novel targets. Execution and
retained branch supports remain unchanged. Draw metrics are averaged within
tree; trees are the independent analysis units. Empty draws are omitted. The
aggregate includes only trees passing the frozen eight-draw analyzability
threshold.

The primary efficacy conjunction requires:

- mean depth-three Brier is at least 1% lower than depth-two;
- a 20,000-sample tree bootstrap 95% interval for paired
  `depth_three - depth_two` Brier has upper endpoint below zero;
- depth three has at least 20 tree wins;
- mean Hamming error does not increase;
- mean exact-extension coverage does not decrease; and
- the mean Brier difference is negative in each 32-tree source study.

Bootstrap seed: `37291`.

## Interpretation

- **Positive irreducibility audit:** bank mechanics, support novelty, endpoint
  power, and every efficacy gate pass.
- **Mechanism-only:** support and endpoint novelty gates pass but the efficacy
  conjunction fails.
- **Inconclusive:** the broad bank leaves insufficient grammar-novel endpoint
  power.
- **Negative:** mechanics are valid but the support-novelty gate fails, or
  powered grammar-novel targets do not show the frozen non-myopic benefit.

No threshold repair, grammar narrowing, target dropping below the frozen
analyzability rule, root reselection, or paid replication is allowed after this
audit is opened. A paid follow-up is authorized by this route only after a
positive irreducibility audit.
