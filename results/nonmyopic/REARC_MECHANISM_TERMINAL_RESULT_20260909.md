# Mechanism qualification stopped on schema/parser mismatch

One launch from frozen implementation9ce813b1, one accepted Luna-medium call,
$0.004554, zero uncertain exposure. Normal stop,1034reasoning tokens,1472total
completion tokens. The run ended failed_closed at0_contrasting_plan; no programs
were compiled, no other tasks/arms ran, and all60withheld outputs stayed sealed.
Exact saved failure replay passes with one recorded call and zero new calls.

The cause is our interface mismatch, not a semantic-model or planner result.
API JSON schema declares description maxLength512characters; local parse_plan
requires at most512UTF-8bytes. Response lengths (characters,bytes) were
(463,465),(476,478),(450,452),(512,514). The final description contained a Unicode
em dash and was rejected at514bytes despite satisfying the declared character
limit. Its text also ends mid-example; normal stop/schema acceptance alone is
not proof of a semantically complete explanation.

The25prelaunch tests missed valid-Unicode boundary cases. New discrepancy tests
also identify two other local-only requirements: array enums do not enforce the
ordered unique p0..p3IDs; minLength1 does not enforce nonblank text. These are
interface contract defects/extra semantic checks, not evidence that the tested
provider violates its declared schema. No result or threshold is repaired here.

Next: give the successor one authoritative plan schema, ideally fixed named
fields so ID order and uniqueness are structural; use character limits locally
and a separate total serialized-byte transport cap. Separate schema validity
from substantive explanation checks. Require adversarial Unicode/order/length
parity checks before source selection or calls. Preserve this run and parser
unchanged for replay; no truncation, normalization or paid retry on this cohort.

This failure means the research comparison was not reached. It neither confirms
nor refutes contrasting mechanisms, predictive coverage or non-myopic efficacy.
Do not start a depth sweep or claim the prior gate has been rescued.

Authenticated credits/usage/balance245/221.517287569/23.482712431.
London Sep9 posted spend1.05900909; conservative recorded spend1.09900909;
remaining allowance3.90099091. No extra call was made to use the remainder.
Raw bank: results/nonmyopic/rearc_mechanism_qualification_20260909.
Previous/current goal turns made progress, but this turn's evidence is an
engineering failure, not progress toward a positive scientific finding. Goal
remains unachieved; fix contract parity before another paid qualification.
