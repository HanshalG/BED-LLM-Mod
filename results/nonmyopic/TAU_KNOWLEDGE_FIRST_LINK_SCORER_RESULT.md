# tau-Knowledge First-Link Scorer V1 Result

## Decision

The scorer V1 serving smoke failed closed after its two myopic calls, before
either non-myopic call. No confirmation task was touched, and this interface is
closed without response coercion or reissue.

## Failure

Both GPT-5.4 responses were complete flat JSON with five distinct, semantically
sensible root scores and rationales. However, every score was encoded as a
canonical digit string such as `"92"` rather than the JSON number required by
the frozen parser. The first parse therefore failed on the numeric type check.

This is a serving-interface failure, not evidence for or against non-myopic
ranking fidelity:

- exactly two physical calls were issued;
- no non-myopic scorer call was issued;
- reasoning tokens, retries, and forced exits were all zero;
- cost was `$0.01935750`; and
- all 20 sealed confirmation tasks remain untouched.

Public failure artifact SHA-256:
`4a649054b08ef32bb56de6a14b59bedfcdc4102260ea6da828b612d9b11ebdd0`.

Private raw checkpoint SHA-256:
`f793d8d21020f4b67416ea8bd9c2a28244685bd3007aa31d916f1f8f3ed95da4`.

## Next interface

A distinct V2 may explicitly require canonical digit strings for scores and
parse only those strings. Its mechanics smoke must use the first two public V2
opportunity records rather than reissuing these responses. Scientific scorer
views, target blindness, first-link endpoint, confirmation split, and efficacy
thresholds must remain unchanged and be recommitted before calls.

## Budget

The project ledger is `$71.12126216` spent with `$34.26354053` headroom under
the pre-Monday ceiling. The `$25` reserve remains protected and OatML was not
used.
