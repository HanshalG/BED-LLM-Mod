# SWE-smith Debug-BED native-amd64 preflight protocol

Date frozen: 2026-08-13

Status: **prospective public-image runner preflight; zero task payloads**

## Purpose

Establish whether a GitHub-hosted Linux runner can execute native amd64
containers before selecting a new SWE-smith mechanics cohort. This is a new
infrastructure preflight. It does not repair, migrate, or reopen the exact
eight-task mechanics cohort closed by the local ARM host.

## Frozen runner and controls

- Repository: `HanshalG/BED-LLM-Mod`
- Branch: `codex/location-finding-llmstrategy`
- Workflow trigger: push to `codex/location-finding-llmstrategy` changing only
  the bound preflight protocol, workflow, implementation, or test path
- Runner label: `ubuntu-24.04`
- Required host values: `uname -m == x86_64`, Docker server OS `linux`, Docker
  server architecture `x86_64`
- Alpine control: `alpine:3.20@sha256:c64c687cbea9300178b30c95835354e34c4e4febc4badfe27102879de0483b5e`
- Ubuntu control: `ubuntu:22.04@sha256:0199853f6d6b20b0424f3c5694a72a62764f01e6a771b1eb48a4197848986c7e`

Each control must run twice in a fresh container and emit exactly its frozen
literal nonce plus `x86_64`. Exit status must be zero. The verifier accepts no
extra field, reordered arm, noncanonical architecture, image tag in place of
the digest, or partial result.

## Privacy and ordering

The preflight may check out only this repository and pull the two public
control images. It must not clone DebugGym, download SWE-smith shards, read an
instance ID, task payload, patch, test, image name, repository source, prior
private mechanics file, or endpoint. It makes no model or OpenRouter call.

The workflow emits a single aggregate JSON artifact. A passing result
authorizes only the prospective freezing of a distinct source-V3 split and
execution-mechanics protocol. It does not authorize selecting or opening a
new cohort until that protocol is committed and pushed.

## Frozen gates

1. Exact protocol, verifier, workflow, and pushed commit bindings reproduce.
2. Workflow event is `push` on the exact bound branch and commit.
3. Runner and Docker host are native amd64.
4. Both digest-pinned images execute exactly twice with canonical outputs.
5. Result reports zero task payloads, model calls, OpenRouter cost, and OATML
   cluster use.
6. No private or identifying SWE-smith value is serialized.

The workflow may use its ephemeral repository token only to commit the
aggregate result under `results/nonmyopic/swesmith_debug_bed_native_amd64_preflight/`.
The result-only commit does not match the trigger paths. Any failure closes
this exact preflight attempt. A retry requires a new prospective protocol and
pushed commit; workflow re-run is forbidden.

## Accounting

- OpenRouter calls: `0`
- OpenRouter cost: `$0`
- OATML cluster use: none
