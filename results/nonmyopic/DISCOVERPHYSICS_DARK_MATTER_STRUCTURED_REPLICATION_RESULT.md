# DiscoverPhysics Structured Fresh-Tree Replication V2 Result

## Status

`FAILED_CLOSED_AT_DISCARDED_PREFLIGHT`

V2 is closed. It produced no model response and opened no scientific tree or
physical endpoint.

## Frozen Run

- Protocol commit: `5639c27`
- Interface:
  `discoverphysics-dark-matter-structured-replication-1`
- Run:
  `discoverphysics-dark-matter-structured-replication-20260728T060000Z`
- Public failure SHA256:
  `d6ad93b21b1301886a70298a7b5cec3e4fd964fe1a4a02613edc0599b0a729cb`

The pinned official simulator initially lacked its local JAX dependency.
Installing JAX `0.11.0` and importing `NBodyDarkMatterExecutor` resolved that
before any adapter was constructed or request attempted. The output directory
was empty before the frozen command was rerun.

## Observed Failure

The sole discarded strict-schema preflight returned an uncharged OpenRouter
HTTP `404`:

> No endpoints found that can handle the requested parameters.

Accounting:

| Quantity | Value |
|---|---:|
| HTTP attempts | 1 |
| Accepted adapter requests | 0 |
| Responses | 0 |
| Prompt tokens | 0 |
| Completion tokens | 0 |
| Reasoning tokens | 0 |
| Retries | 0 |
| Forced exits/finals | 0 |
| Cost | `$0` |
| Scientific support calls | 0/9 |
| Simulator endpoint calls | 0 |

No `RAW_RESPONSES.json`, `MODEL_FROZEN.json`, `POLICY.json`, or endpoint
result exists. Seeds `24700--24717` remain unopened.

## Cause

The live endpoint inventory lists five GPT-5.4 routes, all advertising
`response_format` and `structured_outputs`. The V2 request nevertheless set
OpenRouter's `provider.require_parameters=true` while the inherited generic
payload also included `temperature`, `top_p`, and `top_k`. The same endpoint
inventory does not advertise those sampling fields.

OpenRouter documents that `require_parameters=true` removes any provider that
does not support every parameter in a request; with default routing, a
provider may instead ignore unrelated unsupported parameters. The evidence is
therefore consistent with provider filtering before structured-output
execution, not with a rejected hypothesis schema or malformed model response.

## Decision

Do not rerun or alter V2. Its discarded preflight correctly closed the exact
route before cost or scientific exposure.

A prospectively frozen V3 may change only provider routing to the documented
default while retaining:

- the exact GPT-5.4 model, nonreasoning setting, prompts, strict schemas,
  parser, compiler, likelihoods, support mixture, controls, seeds, thresholds,
  and request counts;
- a new discarded route preflight before any scientific response; and
- fail-closed behavior with no semantic repair or retry.

This is a transport correction on an untouched experiment, not a rescore or
semantic rerun.

## Accounting

- OpenRouter cost: `$0`
- Authenticated balance after V2:
  `$8.990154844`
- Reserve: none
- OatML, Slurm, SSH, or cluster use: `0`
