# DiscoverPhysics Dark-Matter Semantic Belief Smoke Result

## Decision

**The exact ten-call smoke passed serving and support-dynamics gates but
failed the scientific first-link gate. No actual-observation policy run is
authorized under this interface.**

The run completed for `$0.096155` with 10/10 requests, zero retries, zero
reasoning tokens, zero forced exits, and no parser failure. The failure is
therefore about the LLM-derived likelihood and pathway ordering, not
transport.

## Selection

| Root | Probe | LLM immediate EIG | Expected readiness | Blinded pathway score |
|---|---|---:|---:|---:|
| A | northwest target | `.1514` | `75.16` | 75 |
| B | center scout | `.2021` | `73.90` | 74 |
| C | southwest target | `.3428` | `75.74` | 76 |
| D | northeast target | `.2349` | `80.20` | 80 |

The preregistered oracle-qualified ordering was:

- myopic: northeast target D;
- complete path: center scout B.

The LLM-derived tree instead selected southwest C myopically and northeast D
for the complete path. Center sacrificed `.03274` immediate nats relative to
northeast, satisfying that narrow gate, but its pathway score was six points
lower rather than at least five points higher.

## What Worked

- All 8/8 branch refreshes changed the initial semantic support.
- All 4/4 roots produced branch-distinct refreshed supports.
- The central branches chose different continuations.
- Expected-readiness values and final scores varied with unique maxima.
- The eight initial hypotheses represented northeast, northwest, southwest,
  and southeast halos with compact and elongated geometries.

This is genuine path-dependent semantic belief generation, not support
collapse.

## What Failed

The model's qualitative likelihoods overvalued a southwest local test despite
the `.20` southwest prior and undervalued the `.40` northeast target. Its
complete-path scorer then largely followed self-estimated local readiness:
northeast received `80`, southwest `76`, northwest `75`, and center `74`.
The model did not recognize the center probe's externally verified
scout-then-target value.

One central branch selected a due-south continuation on an axis, so the
frozen distinct-quadrant mechanics gate also failed. This is secondary to
the two root-order failures.

## Consequence

The exact free-form qualitative-likelihood and self-scored-pathway interface
closes without prompt repair, alternate seed, reasoning, or model rerun.
The structural oracle result remains valid, but this smoke confirms that an
ungrounded LLM simulator does not preserve its value ordering.

The justified successor is scientifically different: the LLM may generate
an open set of executable hidden-halo hypotheses, while the official
simulator computes likelihoods and trajectory risk for those generated
hypotheses. That keeps semantic support generation load-bearing but removes
the failed self-referential likelihood/scorer layer. It requires a new
preregistration before calls.

Artifacts:

- Preregistration:
  `results/nonmyopic/DISCOVERPHYSICS_DARK_MATTER_SEMANTIC_SMOKE_PREREGISTRATION.md`
- Public smoke:
  `results/nonmyopic/discoverphysics_dark_matter_semantic_smoke/discoverphysics-dark-matter-semantic-smoke-20260727T000000Z/SMOKE.json`
- Public smoke SHA-256:
  `d26fa50b637850d0240e1011eaa47aa53df7d86bb954a5bcb23b47dce6a08ee1`
- Private raw-response SHA-256:
  `9faba6dc8194a6e1542ade783b6757e1bcaf3d321b2f440d21b93f1ef9a0ef6b`
- Initial compatibility preflight: zero HTTP requests and `$0`
- Scientific run: 10 requests and `$0.096155`
- Simulator calls: `0`
- OatML use: none
