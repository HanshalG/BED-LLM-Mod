# ICAE-Bench Value-Blind Release Manifest

Date: 2026-07-29

## Purpose

Freeze the official ICAE-Bench source and a language-stratified task split
before any task-level opportunity score, policy score, model response, or
executable endpoint is used.

ICAE-Bench is a candidate LLM-native sequential BED environment because the
policy asks free-form clarification questions about a fuzzy software
requirement, an official hidden-data Oracle answers only semantically matched
questions, and final utility is measured by released black-box tests.

## Bound Source

- repository: `https://github.com/ALEX-nlp/ICAE-EVAL`
- commit: `66bbabb20a2138d066ac7d6f7ba6768b57c2f79b`
- official gated PRD bundle:
  `https://zenodo.org/records/21639512/files/icae_prd_bundle.tar.gz`
- expected bundle SHA-256:
  `b054e8f03b3c434ffaec3c4ee6cf712d3f25e5bc7cd7ef9eb0a913026fd827d7`

## Split

Within each of the 12 released languages, order the 40 opaque aliases by
`SHA256("50000:<language>:<alias>")`, breaking ties by alias. Assign:

- first 1 to mechanics;
- next 3 to development;
- next 4 to confirmation; and
- final 32 to retained.

This yields 12 mechanics, 36 development, 48 confirmation, and 384 retained
tasks.

The public manifest may contain only aliases, languages, relative artifact
paths, and hashes. It must not emit repository identities, PRD text, hidden
constraints, trigger phrases, Oracle replies, test values, or policy scores.

## Consequence

A valid manifest authorizes opening only the 12 mechanics records for a
zero-call source/opportunity audit. It does not authorize an OpenRouter call.

Any later paid gate must be separately frozen. The scientific endpoint must
ultimately use released executable test utility; constraint recall alone is
insufficient for an efficacy claim.
