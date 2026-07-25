# 2WikiMultiHopQA Directional Chain Audit Result

Date completed: 2026-07-25

## Decision

**Gate failed. Close this 2Wiki construction before any LLM call.**

The dataset contains many genuine directional two-document evidence chains, but
the setup root is always stated in the question on the frozen opportunity
sample. Standard retrieval therefore solves the first decision too often for
this to be a useful non-myopic semantic-planning environment.

## Frozen Sample

- Official April 7, 2021 `data_ids` archive SHA-256:
  `95df2bf56fdabe034e27aebc580e02264232203cf52552f9efe8a919e5529eef`.
- Official `train.json` SHA-256:
  `b318dbafbfed51a8029718fa59be8b616600cbff675a3b587694b28c5eedfc13`.
- Selection seed: `24360`.
- Opportunity rows: 500 (471 compositional, 29 inference).
- Development endpoints accessed: 0/100.
- Holdout endpoints accessed: 0/1,000.
- OpenRouter calls: 0.
- OatML jobs: 0.

All source, count, split-hash, and split-type checks reproduced.

## Results

| Metric | Frozen gate | Observed | Pass |
|---|---:|---:|---:|
| Exact two-triple evidence chains | >=450/500 | 438/500 | No |
| Strict directional chains | >=300/500 | 409/500 | Yes |
| Strict chains with coverage gap exactly one | all | 409/409 | Yes |
| Setup root not named in question | >=50 | 0 | No |
| Title-BM25 setup-root misses | >=75 | 29 | No |
| Paragraph-BM25 setup-root misses | >=75 | 74 | No |

Among the 409 strict chains:

- title BM25 ranked the setup root first on 380, a distractor on 29, and the
  answer child on 0;
- paragraph BM25 ranked the setup root first on 335, a distractor on 71, and
  the answer child on 3;
- every question contained the normalized setup entity or a setup-title alias.

The 91 non-strict rows were excluded for:

- evidence entity/support-title mapping ambiguity: 52;
- answer not unique to child supporting sentences: 18;
- invalid official supporting-sentence reference: 10;
- evidence triples not forming the frozen ordered chain: 10;
- reverse child-to-root link: 1.

## Interpretation

The positive structural count is not enough. These examples genuinely require
combining two documents to produce the answer, and setup-first has the expected
two-action support-coverage advantage. However, the template names the setup
entity directly in every qualifying question. A title-only BM25 baseline
retrieves that root in 92.9% of strict chains.

This is multihop answer reasoning, not a difficult non-myopic first action. An
LLM lookahead policy could appear successful merely by identifying the named
root, while a classical retrieval baseline would already do the load-bearing
work. That would repeat the exact failure mode the project is avoiding.

No development smoke, scorer, or policy comparison is authorized. The sealed
development and holdout rows remain unused.

## Verification

- Focused harness tests: `9 passed`.
- Independent artifact replay recomputed the strict count, both BM25 miss
  counts, unnamed-root count, coverage gaps, and zero-access/call accounting.
- Audit artifact SHA-256:
  `45e8ee45205f516eb2782c8d30bf565a6c5978c1c014c746f222d238e15b5d0f`.

Artifacts:

- `results/nonmyopic/2WIKI_DIRECTIONAL_CHAIN_PREREGISTRATION.md`
- `scripts/twowiki_directional_chain_audit.py`
- `tests/test_twowiki_directional_chain_audit.py`
- `results/nonmyopic/2wiki_directional_chain_opportunity/AUDIT.json`
