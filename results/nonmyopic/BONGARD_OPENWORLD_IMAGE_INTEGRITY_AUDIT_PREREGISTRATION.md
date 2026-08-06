# Bongard-OpenWorld Image Integrity Audit

Date frozen: 2026-08-06

## Purpose

Bind the official Bongard-OpenWorld image archive and verify that the frozen
sequential BED mechanics tasks can be served without corrupt images or
label-bearing filenames entering model-facing payloads.

This audit uses zero model calls and opens no scientific endpoint. It cannot
establish a non-myopic advantage or authorize a paid run.

## Source-Binding Rule

The publisher provides a stable official Google Drive file and exact metadata,
but no archive checksum. The first complete download may establish the
canonical archive SHA-256 only if all gates below pass conjunctively. The
download must come from the URL and exact 5,124,375,111-byte object already
bound by the source/protocol audit. If any structural, CRC, or decode gate
fails, no canonical hash is banked.

All later uses must match the banked whole-archive SHA-256 exactly.

## Frozen Scope

- Validate the ZIP against all 14,140 image references in the bound official
  metadata.
- Stream a full ZIP CRC check without unpacking the archive.
- Decode all 56 images belonging to the four frozen validation mechanics
  tasks directly from ZIP streams.
- Emit mechanics records only under the source audit's opaque task and image
  IDs. Do not emit source UIDs, member names, `pos`/`neg` strings, concepts,
  captions, source positions, candidate labels, or endpoint labels.
- Do not decode or summarize validation development/confirmation/reserve or
  official test image content in the public result.

## Gates

All are conjunctive:

1. The source commit, tree, metadata hashes, and source/protocol manifest hash
   still match their frozen values.
2. The archive size is exactly 5,124,375,111 bytes.
3. The ZIP opens successfully and contains no duplicate, absolute, parent-
   traversing, or backslash-based member paths.
4. The set of non-directory members equals the 14,140 metadata image paths
   exactly: no missing or extra files.
5. Every expected member has positive compressed and uncompressed size.
6. A complete `ZipFile.testzip()` scan returns no bad member.
7. The frozen mechanics partition still contains exactly four tasks and 56
   unique images under the source audit's deterministic split/layout.
8. All 56 mechanics images pass Pillow `verify()` and full pixel `load()`.
9. Every decoded mechanics image has positive width and height and an accepted
   raster format (`JPEG`, `PNG`, `GIF`, `WEBP`, `BMP`, or `TIFF`).
10. The public result contains only opaque mechanics identifiers and image
    integrity statistics; a recursive forbidden-string scan finds no source
    UID, source path, label token, concept, or caption.

## Interpretation

A pass banks the archive SHA-256 and establishes
`image_integrity_status=pass`. It leaves
`scientific_opportunity_status=untested` and `authorizes_paid_calls=false`.
The next admissible step remains a four-task validation mechanics smoke after
the already frozen OpenRouter daily sequence.
