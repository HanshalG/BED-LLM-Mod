# Exact source-prior probabilities for discovered programs

This zero-call numerical addition follows the last turn's completed scoring
mechanics. The proposal-only sequencing question remains unanswered; no paid
permission or revised experiment order is inferred from goal continuation.

## Implemented calculation

`environments/program_induction/prior.py` computes exact rational probabilities
under the frozen source sampler's syntax law. Length2,3,4 has probability1/3.
At each statement the sampler chooses uniformly from the finite type-valid
operation/lambda/argument tuples, subject to consuming the previous result after
the first statement. Consequently:

    P(program syntax) = (1/3) * product_i (1 / valid_choice_count_i).

Choice counts depend on the program's preceding variable types. They do not
depend on observations, execution results or target outputs. The helper rejects
out-of-prior lengths, variable names, types and predecessor violations. It
normalizes these probabilities on distinct supplied syntax when requested;
repeated copies of a program cannot increase its mass.

This is NOT semantic-equivalence-class probability. Nor does it assign source
prior mass to arbitrary CrossBeam trees, which have a different representation
and may lie outside the source prior. Such a conversion needs explicit validation,
not silently treating every found expression as an equally likely source draw.

## Constructed check

With the two initial list variables, there are80 first-statement choices. After
a list result there are55 valid next choices; after an integer result there are6.
Thus these two length2 programs have probabilities1/13200 and1/1440:

    x2 = Reverse x0; x3 = Head x2
    x2 = Head x0;    x3 = Access x2 x0

Both return0 on x0=[0], but they disagree on x0=[0,1]. Restricting the source
prior to these two programs gives weights6/61 and55/61, not1/2 each. This is a
hand-built mathematical fixture, not an observed model proposal or evidence
that prior-weighted predictions improve the benchmark.

## Verification and scope

30 focused prior/prediction/proposal tests passed in1.38 seconds; lint passed.
The new tests independently check the explicit choice counts and probabilities,
reconstruct100 non-scientific-seed sampler paths against the unchanged original
implementation, check duplicates, and reject out-of-prior programs.

The existing uniform-pool forecast format and all banked results are unchanged.
This helper supplies a source-law weighting option for a future explicitly
versioned predictor. It is not silently wired into old experiments or described
as their posterior. Candidates must also satisfy the complete real history
before restricted-prior weights can be interpreted as conditional-pool posterior
weights. Missing-program mass, data-dependent search selection and calibration
remain unresolved.

No new scientific programs, outcomes or model responses were sampled. Credits/
usage245/220.376693994 remain consistent with the London Sept8 ledger and zero
spend. No cluster, protected runtime or automation changes. The full goal remains
incomplete; further lookahead and LLM efficacy are not established by these tests.
