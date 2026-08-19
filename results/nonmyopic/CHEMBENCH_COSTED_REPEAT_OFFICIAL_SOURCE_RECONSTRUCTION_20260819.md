# ChemBench costed-repeat official source reconstruction

Date frozen: 2026-08-19

The metadata-only recovery protocol failed closed before evaluation: temporary
cleanup removed the three frozen source files and the pack containing their blob
objects, not only Git administrative metadata. The surviving commit and tree
objects prove the expected identity, but the original file hashes cannot be
recomputed locally. No replacement medium or hard response, policy, metric, or
gate has opened.

Prospectively authorize one clean reconstruction from the official primary
repository `https://github.com/scientific-discovery/LLM-AutoSciLab`:

1. preserve the damaged directory under a timestamped failure name, including
   its surviving pack/index hashes and failed status;
2. clone into a fresh temporary sibling directory without reusing any working
   file from the damaged checkout;
3. detach exactly at commit
   `acf160eb6c96897748dd92b152703b59b74efc05`;
4. require tree `e042d418fc30c6c70f6d1c6b0636d43f0c1c0f7a`;
5. require SHA-256 values
   `eba9514d68573b4aa8c6a427606f1d0d7a9c431d8396e69ef4ae0774d2a449de`,
   `0dfd47c1858efacb732f0fbceace3ebd61bb30f864ec777d628fb70f876343f3`,
   and `defc6c0c5edafe75dffaa366a61298856f413bd465a6e4e3636e9b0c57124003`
   for the three source paths bound by `verify_source`;
6. run the original `verify_source` unchanged on the reconstructed directory;
7. only after all checks pass, rename the fresh directory to the original frozen
   path `/private/tmp/LLM-AutoSciLab-acf160e`, preserving the source-root string
   already bound by legacy shards; and
8. rerun the unchanged read-only disk-runtime preflight before evaluation.

The network fetch provides bytes only. Commit, tree, and file-hash verification
remain the authority. Any fetch failure, wrong object, dirty checkout, hash
mismatch, or path collision closes reconstruction. This changes no scientific
source, response, policy, control, or gate and authorizes no model/API call.
