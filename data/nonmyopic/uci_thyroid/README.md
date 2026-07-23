# UCI Thyroid Disease Acquisition Files

These files are the `ann-thyroid` cohort and released acquisition metadata from
the UCI Thyroid Disease archive, dataset DOI `10.24432/C5D010`, downloaded from
`https://archive.ics.uci.edu/static/public/102/thyroid+disease.zip` on
2026-07-23. The dataset is licensed CC BY 4.0.

Frozen SHA-256 hashes:

- archive: `a0982569a7442c03a20815db58f271245e7a111b10ac46f6c6b5fa6feee4c1f4`
- `ann-train.data`: `3da53a156bda36cb0c97e9f4b6b111c9226c54c4aa00230de5604b787c47e3a6`
- `ann-test.data`: `c649ea19416e78c7996cfaaa2a9e281cb597d4b075aaa68c494fc3e4ee3aa30b`
- `costs/ann-thyroid.group`: `97b709a689cf3818c91df2eafe95552ad257f142a931524e18ac44cfabfc959a`
- `costs/ann-thyroid.delay`: `1e4594e53f9c42797f3f6e5818fe7f13b31bee808ea14fa6df96b382a840871f`
- `costs/ann-thyroid.expense`: `a0b3ab6a3b5743952f2bc061d3909edc9dacbd8c6b2c286261f094c70665ceac`

The released delay file marks the 16 history/demographic variables as immediate
and TSH, T3, TT4, and T4U as delayed. The group file places those four assays in
one blood-test group, and the expense file assigns a shared collection discount.
The BED adaptation uses one zero-information blood-collection action to unlock
those four assays. FTI remains in the source rows but is not an action because it
is absent from the released cost/delay files.
