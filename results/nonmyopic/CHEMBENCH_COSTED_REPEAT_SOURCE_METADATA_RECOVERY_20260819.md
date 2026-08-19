# ChemBench costed-repeat source metadata recovery

Date frozen: 2026-08-19

The read-only disk-runtime preflight stopped before evaluation because temporary
directory cleanup removed the pinned source checkout's Git administrative files:
`.git/config`, `HEAD`, refs, and index are absent. The existing object pack,
working source files, and frozen scientific paths remain present. No replacement
medium or hard response, policy, metric, or gate has opened.

Recover Git metadata without a network call or source-file checkout:

1. hash and retain the surviving pack and index files;
2. run `git init` only to recreate administrative structure;
3. require the existing object database to contain commit
   `acf160eb6c96897748dd92b152703b59b74efc05`;
4. point a new local `frozen-costed-repeat` branch and `HEAD` at that exact commit;
5. populate only the Git index with `git reset --mixed HEAD`, never `--hard`;
6. require the working source paths to remain byte-identical; and
7. rerun the original fail-closed `verify_source`, requiring commit
   `acf160eb6c96897748dd92b152703b59b74efc05`, tree
   `e042d418fc30c6c70f6d1c6b0636d43f0c1c0f7a`, and all three frozen file
   SHA-256 values.

Any absent commit object, tree mismatch, source hash mismatch, checkout-induced
change, or network requirement closes this local recovery and authorizes no
replacement run. Git metadata reconstruction changes no scientific source and
does not relax the original source gate.
