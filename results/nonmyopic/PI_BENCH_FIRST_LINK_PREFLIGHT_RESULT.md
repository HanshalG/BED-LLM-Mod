# Pi-Bench Dynamic-Support Preflight Result

Date: 2026-07-28

**Status: interface v6 passed; the frozen five-task serving gate is authorized.**

## Scope

The preflight used only `Financier_task_001` from the already-open mechanics
partition. It executed the complete target-blind planning path:

1. eight latent requirement-set worlds and six root questions;
2. initial semantic mapping;
3. four common rollout worlds per root;
4. support regeneration for every incomplete branch;
5. follow-up semantic mapping and immediate-versus-terminal root selection.

It did not execute private true-intent transitions, naive thinking, or any development
task.

## Passed Interface

- run: `pi-bench-planning-preflight-v7-20260728T002200Z`;
- implementation commit: `8e6ed46`;
- interface: `pi_bench_dynamic_support_v6`;
- source commit: `383910b1698758a198b86037c63a111c8edc32ad`;
- initial worlds: `8/8`;
- root questions: `6/6`;
- incomplete rollout branches: `24/24`;
- unique refreshed-support fingerprints: `24/24`;
- semantic pairs: `528`;
- padding-normalized pairs: `0`;
- padding-normalization rate: `0.000`;
- myopic root: `Q6`;
- depth-two root: `Q1`;
- root choice changed: yes.

The root disagreement is an opportunity diagnostic, not an endpoint result. No true
hidden intent was used in either score.

## Usage

- physical requests: `4`;
- HTTP attempts: `4`;
- transport retries: `0`;
- prompt tokens: `64,998`;
- completion tokens: `9,925`;
- reasoning tokens: `0`;
- forced exits: `0`;
- cost: `$0.311370`.

Authenticated OpenRouter balance after the preflight: `$27.088582094`.

## Failed-Closed Development History

Six earlier mechanics-only interface attempts were retained privately:

- generic adapter routing 404: zero accepted requests, `$0`;
- accepted failed-closed requests across interfaces v1-v5: `19` requests,
  `$1.2113275`;
- failure causes, in order: variable boolean-vector length, an overbroad generic
  question regex, zero/one-based index ambiguity, duplicate matched IDs, and
  nonzero fixed-width padding.

Each failure occurred before any endpoint or development model call. The final v6
representation uses fixed seven-bit local maps, logs deterministic absent-padding
masking, and fails the formal gate above a 10% normalized-pair rate. The successful
preflight required no masking.

Public artifact:
`results/nonmyopic/pi_bench_first_link_preflight/pi-bench-planning-preflight-v7-20260728T002200Z/PREFLIGHT.json`.
Raw prompts and responses remain untracked.
