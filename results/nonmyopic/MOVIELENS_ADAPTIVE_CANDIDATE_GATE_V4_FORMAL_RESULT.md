# MovieLens Adaptive-Candidate v4 Formal Result

Date: 2026-07-24

Status: failed the frozen sensitivity screen. No candidate outcome, held-out rating,
branch refresh, scorer, policy, or depth run was evaluated.

The 12 fresh formal users completed exactly 12 initial profile generations and 12
profile-only likelihood requests. The process then applied the preregistered futility
stop:

- mean best-of-16 EIG: `0.03270` versus `0.02` required, pass;
- users with best-of-16 EIG at least `0.02`: `4/12` versus `8/12` required, fail;
- exactly 24 requests and zero reasoning tokens;
- total cost `$0.17247900`;
- remaining balance `$20.881994843`.

The four responsive users had maximum EIGs `0.06647`, `0.04090`, `0.14065`, and
`0.08019`. The other eight ranged from `0.00096` to `0.01823`. Adaptive candidate
search therefore raises the upper tail but does not make natural semantic-profile
disagreement broadly reliable.

The process stopped before reading source outcomes. Do not run branches on the four
responsive users as a post-hoc subset. The exact v4 apparatus is closed. A distinct
future design could prospectively define high semantic uncertainty as an enrollment
criterion on a new population, but must freeze that conditional claim and sample
before any outcomes.

Artifacts:

- `results/nonmyopic/movielens_adaptive_candidate_gate_v4/formal_seed24305_20260724/GATE.json`
- `results/nonmyopic/movielens_adaptive_candidate_gate_v4/formal_seed24305_20260724/run.log`
