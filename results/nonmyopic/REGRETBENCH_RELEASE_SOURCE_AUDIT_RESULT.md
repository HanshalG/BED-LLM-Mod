# RegretBench Release Source Audit Result

Date: 2026-07-28

**Status: failed before task-value access and before any model call.**

## Pinned Source

- official repository: `https://github.com/ngocminhta/RegretBench`;
- commit: `5e105c6bf033a7261ab5a7fd6f581dbf2d76790a`;
- detached checkout size: `61 MB`;
- repository state: clean.

## Reproducibility Check

The release documents `OpenDomainQA` as containing:

- `21,252` train CIGs;
- `0` development CIGs;
- `6,286` test CIGs.

Its committed `SHA256SUMS` contains `27,538` entries, exactly the documented
train-plus-test total. Local verification failed:

```text
error: checksum mismatch for .../data/OpenDomainQA
```

The mismatch is not a content-hash discrepancy:

- all `6,286` committed test CIGs match their published hashes;
- `21,252` expected train paths are missing;
- `data/OpenDomainQA/train` does not exist in the Git tree;
- `git ls-tree` contains only `data/OpenDomainQA/test` for split data.

As a control, the same official checker successfully verified all `377`
`ProductRecommendation` CIGs.

## Gate Decision

The first preregistered admission condition required the official release to validate
and reproduce its checksums. It fails. More importantly, the preregistered source-clean
partition required `OpenDomainQA/train`, while the repository warns that its available
`test` split is not pristine. Therefore:

- no metadata manifest can be formed from the frozen source boundary;
- no task prompt, intent description, facet value, reference question, or terminal
  answer was inspected;
- no mechanics or opportunity value was computed;
- no OpenRouter call was made;
- no substitution from the non-pristine test split is allowed.

This result does not judge RegretBench's scientific formulation. It says only that the
newly published commit cannot support the preregistered clean experiment. The route
may be reconsidered only after an official versioned release adds the checksum-listed
train files or provides another genuinely clean source split. It must not be repaired
locally by copying upstream source data or treating the current test set as held out.

OpenRouter spend: `$0`.

