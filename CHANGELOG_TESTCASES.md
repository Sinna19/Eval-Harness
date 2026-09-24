# Test dataset changelog

Every entry here corresponds to a `DATASET_VERSION` bump in `test_cases.py`.
Bump the version whenever `TEST_CASES` changes meaningfully (a case added,
removed, or its expected behavior rewritten) and add a line here explaining
why. `dataset_fingerprint()` (also in `test_cases.py`) is a content hash of
the list and changes automatically with any edit, even a typo fix — it's a
safety net for catching changes nobody remembered to log here, not a
replacement for this file.

This matters for `regression.py`: a baseline saved against one dataset
version and compared against a different one is an apples-to-oranges
comparison (case IDs may not even overlap), so `regression.py` prints both
versions and flags a mismatch explicitly.

## 1.1.0 — 2026-09-24
- No case content changed. Added dataset versioning itself
  (`DATASET_VERSION` + `dataset_fingerprint()`), plus tooling that depends
  on it: `regression.py`, `metrics.py`, `generate_adversarial.py`.

## 1.0.0
- Baseline: the original 16 hand-written cases documented in `README.md`
  (Key Findings #1–9).
