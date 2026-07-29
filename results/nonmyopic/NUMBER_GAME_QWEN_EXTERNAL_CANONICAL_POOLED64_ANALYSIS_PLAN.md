# Number Game Qwen External-Canonical Pooled-64 Analysis Plan

Date frozen: 2026-07-29, after both independent cohort outcomes were open.
This is explicitly retrospective and cannot rescue either source study's
registered composite null.

## Purpose

Quantify reproducibility of the depth-three versus myopic effect across the
two independent fresh Qwen cohorts, while exposing rather than averaging
away the heterogeneous depth-three versus depth-two result.

## Fixed Sources

- Confirmation V1 `RESULT.json` SHA-256
  `370e1c2923e56fb6a8344558db0a69bd5f86a8b013da7d9452675df380f7f12b`.
- Replication V2 `RESULT.json` SHA-256
  `a03c5a6f6e01af403ce27ef7984e29176bf0aa5f40caeae2bb6d213ce8c5dc83`.

Each source contains 32 disjoint Qwen planning trees scored on the same exact
33-concept canonical bank. No model or endpoint calls are made.

## Analysis

- Preserve each source result and transport status.
- Pool equal-weight tree outcomes with a stratified bootstrap that resamples
  32 trees independently within each cohort.
- Report every matched baseline, Hamming, coverage, and ranking diagnostic.
- Descriptive robustness checks require both cohorts to show at least 8%
  Brier gain over myopic, interval below zero, and at least 20 wins; pooled
  gain at least 10%, interval below zero, and at least 40 wins.

These checks summarize already-open evidence. They are not prospective
hypothesis tests and cannot alter either cohort's `gated_null` status.
