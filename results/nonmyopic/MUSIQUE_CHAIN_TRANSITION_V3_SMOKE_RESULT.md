# MuSiQue Indexed Chain-Transition V3 Smoke Result

Date: 2026-07-24
Run: `musique-chain-transition-v3-smoke-20260724T200526Z`
Status: **hash-locked serving recovery passed; formal V3 authorized**.

The first execution made two initial calls and stopped on the documented title-vs-ID
harness bug. The recovery reused the exact checkpoint and issued only 12 missing branch
calls. Combined usage was exactly 14 requests, zero reasoning tokens, zero retries or
forced exits, and `$0.00440975`. The frozen reused checkpoint hash was
`a71110359dab7ffa729bdd3c365fbc5db676baa5aaf6ccf9d35ef9dc292ad15a`.

Both true root document IDs entered their six-action sets and both gold ordered pairs
were initially omitted. One row recovered its gold pair only after the correct root
document was opened; the other recovered under no branch. Immediate EIG selected the
recovering branch, so smoke-only realized regret was zero. These observations are
descriptive and do not alter the frozen formal conjunction.

Artifact:
`results/nonmyopic/musique_chain_transition_v3_smoke/musique-chain-transition-v3-smoke-20260724T200526Z-recovery/SERVING_SMOKE.json`.
