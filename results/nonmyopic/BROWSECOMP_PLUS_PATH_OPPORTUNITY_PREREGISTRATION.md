# BrowseComp-Plus Path-Opportunity Preregistration

## Claim Being Tested

Before building another paid semantic planner, test whether this fixed-corpus
benchmark actually contains observation-enabled evidence paths. The audit
uses the official released BM25 + GPT-5 trajectories as a public action bank,
not as our method or paper result.

A path-compatible bridge event occurs when:

1. a later search query contains at least two content tokens absent from the
   original question but present in documents returned earlier in that
   trajectory; and
2. the later search retrieves at least one previously unseen official
   evidence document.

A gold bridge additionally retrieves a previously unseen answer-bearing
document. A bridge-after-miss requires the immediately preceding search to
have retrieved no new evidence. These definitions establish that the
observation made a useful semantic continuation available; they do not prove
that the released agent causally copied those tokens.

## Frozen Inputs

- Content-blind manifest SHA-256:
  `fd3d0e05f7110974f3312711e24f3d98ac19470447dcd1fccc6dbbf05221a862`.
- Official encrypted BM25 + GPT-5 trajectory SHA-256:
  `74e8e2b24d0ff250ec86a26c392a55a9a5d34ac9911d253b2e35e017341cf944`.
- Split: exact `120` opportunity tasks only.
- Official evidence and gold qrels.
- Only opportunity questions and their released search outputs are decrypted.
  Answers and source document bundles are not loaded.
- Development `40` and holdout `665` remain sealed.

## Conjunctive Gate

The audit passes only if all conditions hold:

- exactly `120` task records;
- at least `110` completed trajectories with at least two searches;
- at least `45` tasks retrieve any official evidence;
- at least `20` tasks contain a bridge-enabled evidence hit;
- at least `8` contain a bridge-enabled gold hit;
- at least `20` first retrieve evidence on search 3 or later;
- at least `10` contain a bridge hit immediately after a no-new-evidence
  search; and
- bridge-evidence tasks span low/mid/high evidence-count bins with at least
  `3/6/5` tasks.

Failure closes BrowseComp-Plus before paid work. Passing authorizes only a
separately committed, sub-`$1` mechanics smoke on the five already open
mechanics tasks. The eventual method must generate semantic answer support,
update that support from returned text, and rank multi-step query strategies;
the released GPT-5 trajectories cannot be used as its policy. No OpenRouter
or OatML call occurs in this audit.
