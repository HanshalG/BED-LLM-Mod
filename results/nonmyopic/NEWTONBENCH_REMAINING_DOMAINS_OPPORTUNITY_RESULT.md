# NewtonBench Remaining-Domains Opportunity Result

Date: 2026-07-28

**Decision: the gate failed. No model call or confirmation-bank evaluation is
authorized.**

## Integrity

- official source commit:
  `912a4ba5f4356ddd06acc16e44460ca30be4abc2`;
- frozen action-manifest SHA-256:
  `7280b2116d7920a5b82b1077b38ca635344ea181286a0383f4b88677656c6f51`;
- audit artifact SHA-256:
  `8259243717aeed15de2017b6fb1f838452e1d87dce65031fd0da064e3965a75a`;
- included cells: `10` domains x `3` declared noise levels = `30`;
- prediction outputs: `2,880 / 2,880` finite;
- OpenRouter calls/cost: `0 / $0`;
- OatML, Slurm, and SSH use: `0`.

Every cell used the frozen 32-action Latin-hypercube development bank, uniform
nine-law prior, official relative Gaussian observation convention, and both 9-node
and 15-node quadrature. Because no development cell passed, the independent
confirmation action bank remained unevaluated as required.

## Results

The table reports 15-node immediate sacrifice and adaptive depth-two gain. A zero
means the same root maximized both objectives.

| Domain | Noise | Greedy root | D2 root | Immediate sacrifice | D2 gain | Stable roots | Pass |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| Gravity | `.1` | 23 | 23 | `0` | `0` | Yes | No |
| Coulomb force | `.1` | 0 | 30 | `.091878` | `.000469` | Yes | No |
| Magnetic force | `.1` | 16 | 16 | `0` | `0` | Yes | No |
| Fourier law | `.1` | 6 | 6 | `0` | `0` | Yes | No |
| Radioactive decay | `.1` | 15 | 15 | `0` | `0` | Yes | No |
| Underdamped harmonic | `.1` | 6 | 10 | `.168678` | `.000327` | No | No |
| Malus law | `.1` | 5 | 0 | `.062306` | `.000080` | No | No |
| Hooke law | `.1` | 30 | 30 | `0` | `0` | Yes | No |
| Bose-Einstein distribution | `.1` | 2 | 2 | `0` | `0` | Yes | No |
| Heat transfer | `.1` | 0 | 0 | `0` | `0` | Yes | No |

All `.0001` and `.01` cells also failed. At those lower noise levels, the same root
was selected by both objectives in every domain. The only low-noise numerical
root-instability flag occurred for underdamped harmonic at `.0001`, where both
reported margins were zero; it cannot create a scientific opportunity.

## Interpretation

The full released vanilla-equation family now has a consistent diagnosis. Together
with the earlier Snell and sound-speed audits:

- many domains reveal essentially the full nine-law identity with one unrestricted
  experiment;
- where a depth-two objective changes the first action, the greedy root followed by
  its own adaptive continuation already captures nearly all available value;
- the largest new terminal advantage is `.000469` nats, about twenty-one times below
  the preregistered `.01`-nat minimum.

This separates iterative experimentation from non-myopic experimental design. Two
measurements can help law discovery without making the first measurement materially
different from greedy EIG.

No noise interpolation, action restriction, hypothesis pruning, domain substitution,
or runner-up confirmation is allowed. NewtonBench remains useful as an interactive
LLM scientific-discovery benchmark, but its released unrestricted vanilla-equation
tasks do not provide the native planning tradeoff required for the headline claim.

## Artifact

The complete per-action values, quadrature diagnostics, continuation selections, and
prediction matrices are stored in:

`results/nonmyopic/newtonbench_remaining_domains_opportunity/AUDIT.json`

