# SWE-smith Debug-BED source V3 native-amd64 protocol

Date frozen: 2026-08-13

Status: **prospective metadata-only fresh split; zero model calls**

## Dependencies

V3 requires:

- the passing source V2 audit;
- the terminal closure of the exact V2 eight-task mechanics cohort;
- native-amd64 preflight commit `5150d0da`, whose independently verified
  result binds GitHub `ubuntu-24.04`, native x86_64 Docker, and two exact arms
  of each digest-pinned public control image.

V3 does not repair or migrate the V2 cohort. It creates a distinct split from
rows that remained in the unopened V2 reserve.

## Frozen metadata selection

1. Reconstruct the exact V2 population and V2 reserve from immutable Parquet
   projections and released DebugGym exclusions/splits.
2. Compute a fresh order over V2 reserve IDs with
   `sha256("swesmith-debug-bed-v3-native-amd64|" + instance_id)`.
3. Group rows by nonempty released `image_name`. For each image, retain only
   composed rows with nonempty F2P and P2P lists.
4. Choose the image with the largest retained row count; break ties by
   `sha256(image_name)`, then the image string.
5. Take the first twelve rows for that image in the fresh V3 order as the
   private structural-screen prefix. The first eight rows in that prefix that
   independently parse to 2--4 nonoverlapping patch hunks become mechanics
   slots 0--7. Fewer than eight closes V3 before execution.

The metadata selector publishes only aggregate counts, ordered ID hashes,
selected-image hash, and immutable source/runtime bindings. It must not publish
an ID, image name, repository, patch, problem, test name, source file, or
endpoint. Structural parsing occurs only after this selector is committed and
pushed.

## Native runtime binding

- Python: 3.12
- `swesmith==0.0.4` wheel SHA-256:
  `149d3fc54bcb0d3b3010121e7de4a13e5e6f31bee08be5918dea1f7b0a8dd5ef`
- Docker registry transformation: DebugGym's `swebench/<image>:latest` maps to
  released `jyangballin/<image with __ replaced by _1776_>:latest`.
- The mechanics protocol must resolve and freeze the selected registry's
  linux/amd64 digest before any task execution.

## Gates

V3 passes only if:

1. all V2 source bindings and gates reproduce;
2. the native-amd64 preflight result and both verifiers reproduce;
3. V3 screening rows are exactly twelve unique V2-reserve rows on one image;
4. V3 rows are disjoint from V2 mechanics/opportunity, effective development,
   effective confirmation, exclusions, and all previously opened task rows;
5. selected rows all have nonempty F2P/P2P metadata and composed IDs;
6. only permitted aggregate/hash metadata is serialized;
7. calls, cost, endpoints, and OATML cluster use remain zero.

A source pass authorizes only a separately frozen structural screen and
zero-model-call mechanics workflow. No task payload or model call opens here.

## Accounting

- OpenRouter calls: `0`
- OpenRouter cost: `$0`
- OATML cluster use: none
