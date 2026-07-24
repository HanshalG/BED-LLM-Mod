# MovieLens Depth-2 Semantic Policy v8 Formal Recovery

Date: 2026-07-24

Status: recovery preregistered; no v8 efficacy metric or source outcome viewed.

The formal run completed all 1,056 model requests with zero reasoning tokens and
cost `$5.00894919`, then failed closed while parsing terminal likelihoods. Its
private raw SHA-256 is
`11f1311024b559008db543d037b518f5e89a35b2d5b800c1b8ce7494ffc87cfb`.

Of 25,600 five-number terminal probability rows, exactly two fail the original
`[.98,1.02]` sum tolerance. Their totals are `.90` and `.94`; all entries are finite
and nonnegative. The other 25,598 terminal rows pass. Initial and first-level
likelihoods already parsed under the original rule.

No candidate outcome or held-out rating was read: the runner failed before policy
path construction, and no source outcome has been inspected manually.

## Frozen Recovery

The recovery will:

1. require the exact private raw hash above;
2. replay the frozen six response batches locally with zero endpoint calls;
3. keep the original parser tolerance for initial and first-level likelihoods;
4. accept terminal rows only when their sum lies in `[.90,1.10]`, then normalize;
5. require exactly two terminal rows to use this expanded tolerance;
6. preserve all policy definitions, users, branches, outcomes, and efficacy gates;
7. verify the original 1,056-request, zero-reasoning, `$5.00894919` accounting.

Any different raw hash, additional repaired row, nonfinite or negative value, schema
error, or probability total outside `[.90,1.10]` fails closed. The final result will
be labeled as a parser-recovered result rather than an untouched formal run.
