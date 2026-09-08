# Conditional-prior rejection feasibility

Prospective new inference diagnostic, not a replay or rescue of either prior
pilot. Preserve the same grammar and input laws. No action optimization, LLM,
larger initial reference bank or changed output alphabet.

Freeze code/protocol before drawing new programs. Verify original pilot artifact
SHA84bdcb025e70d3bbbb3640c5e40f86ae271f005098468f3056637665ba9aa476
and original input hashes. For each of four panels use four new truth programs,
seed6100000+1000*panel+slot. Query input indices0,1,2,3 in fixed order. Test each
history length1..4 separately:64contexts, with no selected easy histories.

For each context, draw up to2048 fresh prior programs, stopping on16 exact
matches to the complete history. Proposal seed stream uses Random with seed
7100000+1000*panel+10*slot+length and independent128-bit generated seeds. Program
generation otherwise uses the original syntax prior. Retain multiplicities;
no truth injection, deduplication, mutation or use of future target outcomes.
Independent prior rejection has the conditional-prior sampling law in the ideal
iid model; pseudorandom implementation and16particles are not an accuracy proof.

Count attempts and history evaluations, full completion, empty/partial sets and
wall time. Partial sets are banked but not scored as completed inference. Do not
interpret accepted/attempts under this stopping rule as an unbiased evidence
estimator. ERROR remains a normal bounded-interpreter observation.

On complete contexts only, fix predictions for the original32 targets before
evaluating their truth outputs. Report conditional-subset half-Brier, internal
risk and target zero-mass count; do not extrapolate them to failed contexts or
claim policy efficacy. Contexts share16truths and are not independent replicates.

Any context failing to obtain16matches establishes that this fixed inference
budget is inadequate there. A full64/64 fill would authorize only further
calibration assessment, never a paid call or a claim of non-myopic benefit.
No post-response enlargement of this experiment's cap or favorable-case rerun.

Ten seconds per context including target scoring; unexpected exceptions or time
exhaustion fail the run with a checkpoint. Reserve output first and checkpoint
each context. No transition database, source dataset or external worker. The
interpreter is the same hash-verified ExeDec source as the initial pilot.
