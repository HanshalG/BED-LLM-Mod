# SWE-bench Lite Graph Unlock V2 Preregistration

## Status

Frozen before reading any SWE-bench Lite test problem statement, patch, changed
file, or test patch. The prior audit read only the test columns
`instance_id`, `repo`, and `base_commit`; this V2 registration does the same
until the split and method are fixed.

This is a zero-call opportunity gate. Failure closes the exact graph-expansion
construction before any LLM use.

## Motivation

The first development audit used code-IDF followups and found one strict
non-myopic file unlock among nine no-direct-leak issues. A target-blind
import/reverse-import expansion, developed only on those same nine disclosed
issues, reaches two:

- the original `pydicom/valuerep.py` unlock; and
- `pydicom/dataelem.py`, reached after inspecting files that import or are
  imported by it.

The new gate asks whether that code-observation structure transfers to the
untouched official test split. It does not reuse the closed IDF continuation,
select the favorable pydicom repository, or expose a test endpoint to an LLM.

## Data and Split

- Dataset: `SWE-bench/SWE-bench_Lite`.
- Revision: `69611d31007e1c6731db8bd5b5c3f2d33f5bab6e`.
- Test Parquet SHA-256:
  `f46f2e3f003f2552932393da4b223e1e0456a2c71eba8b73ae58f29646c1278b`.
- Split seed: `24397`.
- Opportunity ID-list SHA-256:
  `7cbf9eb9d2ebca04fb7568d14cd49dbc2d9cae6797f3a95a646a3793caa0018b`.
- Development ID-list SHA-256:
  `4da3ca0c7ade8a16071d0573a0d5c78091543fa0fa68b17092f2be1fd402aab7`.
- Holdout ID-list SHA-256:
  `2c21db6b128227f92ac7dd158cadbc67be9d4cf10aff2c7c57fb70a1ef5a6ac1`.

The split is generated from metadata columns only. Within each of the 12
repositories, rows are sorted by `instance_id`, shuffled by one seeded RNG,
and assigned:

- first two: opportunity (`24` total);
- next one: development (`12` total);
- all remaining rows: sealed holdout (`264` total).

Only the 24 opportunity rows may have problem and patch columns loaded in this
audit. Development and holdout problem statements, patches, changed-file
endpoints, and tests remain sealed.

## Eligible Endpoint

An opportunity row is eligible when:

- its patch changes exactly one non-test Python file;
- that file exists in the repository at the exact `base_commit`; and
- the issue text does not case-insensitively contain its full path, basename,
  or basename stem of at least five characters.

All exclusions and their reasons are reported. At least 12/24 rows must be
eligible; no replacement rows are drawn.

The external endpoint is binary coverage of the changed file.

## Frozen Retrieval Tree

Each exact-commit repository is indexed over tracked Python files. Documents
contain six repetitions of path tokens plus at most 200,000 source characters.
The five target-blind root-query templates are unchanged from V1:

1. full issue text;
2. first nonempty line;
3. concatenated backtick/code spans;
4. exception and identifier-like tokens; and
5. longest paragraph.

After normalization and deduplication, each root retrieves the top three BM25
files.

V2 replaces only the continuation. Python imports are parsed at the exact base
commit. Candidate followup files receive:

- `+5` for each retrieved root file that imports the candidate;
- `+3` for each retrieved root file imported by the candidate;
- `+1` for each retrieved root file sharing the candidate's directory; and
- `+.25` per distinct issue token appearing in the candidate path.

Candidates with no graph, reverse-graph, or directory relation are excluded.
The top three candidates, tie-broken by path, form the root's one
observation-conditioned followup.

## Values and Controls

- Immediate value: target appears in the root top three.
- Pair value: target appears in root or graph followup.
- Immediate-greedy root: first root attaining maximum immediate value.
- Oracle root: first root attaining maximum pair value.
- Strict non-myopic gap: oracle pair value exceeds the best graph tail under
  the immediate-greedy root.

This is an opportunity upper bound, not a deployable policy result.

## Frozen Gates

All must pass:

- exact metadata split sizes `24/12/264`, disjoint and exhaustive;
- all 24 opportunity rows reproduce at their exact base commits;
- at least 12 eligible no-direct-leak single-file Python issues;
- every eligible issue has at least three distinct roots;
- direct root coverage is at most 80% of eligible issues;
- pair gain occurs on at least `max(4, ceil(.15 * eligible))`;
- strict non-myopic gap occurs on at least
  `max(4, ceil(.15 * eligible))`;
- strict gaps span at least three repositories;
- mean pair gain is at least `.15`; and
- mean strict non-myopic gap is at least `.15`.

Failure closes V2 without threshold, top-k, edge-weight, split, repository, or
eligibility repair.

## Conditional Next Stage

Passage authorizes only a separately preregistered smoke on development rows.
That smoke must:

- use multiple independent LLM proposal batches before deterministic
  deduplication, directly addressing the tau-Knowledge generation-instability
  diagnosis;
- let the LLM generate initial file hypotheses and regenerate them after
  observed source;
- compare path-regenerated non-myopic scoring with myopic, fixed-support,
  BM25, and matched-compute controls;
- keep changed files and patches hidden until all scores freeze; and
- cost no more than `$0.75`.

No test holdout policy run is authorized by this document.

## Mechanics Amendment Before Result

The first audit invocation produced no result artifact and was manually stopped
while reading the first Astropy commit. Its source loader called `git show`
once per Python file, which is prohibitively slow for partial clones.

Before rerunning, source extraction is changed to one `git archive` call over
the exact same `base_commit` and `:(glob)**/*.py` pathspec. The same path
filters and 200,000-character per-file cap are applied after reading the
archive. This changes no source bytes, split, issue, endpoint, query, BM25
document, graph edge, weight, top-k, gate, or API behavior.

## Budget

The opportunity audit uses zero API calls and zero OpenRouter spend. OatML is
not used.
