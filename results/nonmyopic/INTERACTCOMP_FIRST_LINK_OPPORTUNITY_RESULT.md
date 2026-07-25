# InteractComp Regenerated-Support First-Link Result

Date: 2026-07-25

## Outcome

**The first-link gate failed. The exact two-task interface is closed, and no
depth-two efficacy run is authorized from this result.**

The run was mechanically clean and did not exhibit general serving or support
collapse. Both tasks produced eight valid initial particles, four unique
questions, sufficient informative questions and non-unknown true responses,
eight valid refreshed particles at every root, exact-target recovery, and
root-dependent endpoint range.

The failure was ranking robustness. Immediate fixed-support EIG selected the
externally best root on one task and ranked roots inversely on the other.

## Frozen Results

| Metric | Task 76 | Task 38 | Gate |
|---|---:|---:|---:|
| Valid initial particles | 8 | 8 | 8 each |
| Unique questions | 4 | 4 | at least 3 each |
| Questions with EIG >= `.30` | 4 | 2 | at least 2 each |
| Non-unknown true responses | 3 | 3 | at least 2 each |
| Valid particles per refreshed root | 8,8,8,8 | 8,8,8,8 | at least 6 |
| Initial exact-target mass | `.000` | `.000` | descriptive |
| Root endpoint range | `.125` | `.125` | at least `.125` each |
| Selected exact-target mass | `.125` | `.000` | descriptive |
| Oracle exact-target mass | `.125` | `.125` | target recovered each |
| Selected root equals oracle | yes | no | descriptive |
| EIG-endpoint Spearman | `+.8165` | `-.5443` | positive each |
| Selected gain over initial | `+.125` | `.000` | mean at least `.125` |
| Selected gain over candidate mean | `+.09375` | `-.03125` | mean at least `.05` |

Aggregate results:

- mean EIG-endpoint Spearman: `+.13608`, below `.20`;
- mean selected gain over initial: `+.0625`, below `.125`;
- mean selected gain over candidate mean: `+.03125`, below `.05`; and
- mean initial, selected, oracle, and candidate endpoints: `.000`, `.0625`,
  `.125`, and `.03125`.

The per-task positive-correlation gate and all three aggregate ranking/selection
gates failed. Every serving, diversity, refresh, endpoint-range, integrity, and
cost gate passed.

## Mechanism

Task 76 is a clean positive first link: the highest-EIG root also had the only
truth-bearing refreshed support. Its score-endpoint correlation was `+.8165`.

Task 38 exposes the central open-support failure. All eight initial particles
represented the same wrong interpretation. The only truth-bearing refreshed
root asked a consensus-check question for which all eight particles predicted
the same response, so its fixed-support outcome entropy was exactly zero. The
actual closed-mode response contradicted that consensus and caused support
regeneration to recover the exact target with mass `1/8`. Higher-entropy roots
recovered no target mass.

This is not posterior collapse after the query; it is initial support
misspecification. Conventional EIG over a fixed support assigns no value to a
question whose predicted response is identical under every current particle,
even when the surprising response would reveal that the entire support is
wrong. The result therefore does not validate immediate EIG as the first link
for InteractComp.

A distinct future route would have to score the value of branch-conditioned
support expansion itself, including low-probability support-invalidating
answers, rather than use only entropy over classifications of current
particles. It must be developed on these now-open tasks and evaluated
prospectively on fresh encrypted tasks. This result does not authorize a
same-interface rerun, task substitution, or depth-two claim.

## Integrity And Cost

- Preregistered commit: `f562166`.
- Run ID: `interactcomp-first-link-opportunity-20260725T102000Z`.
- Source commit: `9cdf7f804f527ad32a405efaa6c86aae03692556`.
- Models: `openai/gpt-5.4-mini` generator and `openai/gpt-5.4` responder,
  both non-thinking.
- Requests/attempts: `112/112` (`104` generator, `8` responder).
- Retries/reasoning tokens/forced exits: `0/0/0`.
- Cost: `$0.06194525`.
- Private raw SHA-256:
  `fb90c37a65d004a79b1b0aff8dce986aaf0de3ff0abbd633b68aff01b83d3f95`.
- Public artifact:
  `results/nonmyopic/interactcomp_first_link_opportunity/interactcomp-first-link-opportunity-20260725T102000Z/OPPORTUNITY.json`.
- Project-ledger spend after the run: `$86.25126556920749`.
- Monday local allowance remaining: `$14.892039249999925`.
- Authenticated OpenRouter remaining after the run check: `$44.177721634`, or
  `$19.177721634` above the protected `$25` reserve. The provider endpoint
  appeared to lag the adapter ledger, so the stricter ledger value remains
  controlling.
- OatML resources used: none.
