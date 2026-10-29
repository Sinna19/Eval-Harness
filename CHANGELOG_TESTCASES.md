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

## 1.3.0 — 2026-10-10
- No cases added or removed. Three cases changed what they check:
  - `edge_final_sale` -- `must_not_contain` now also forbids "proof of purchase":
    a clearance refusal has no reason to ask for proof, and a response template
    had started adding it.
  - `normal_1` -- gained a deterministic gate (`must_match_any_regex`) requiring a
    conditional word ("if", "unless", ...) in the same sentence as "Final Sale" /
    "clearance". The judge reliably passed replies that asserted the item wasn't
    Final Sale when the customer never said so (rule R12).
  - `mt_shipping_context_followup` -- added a 4th customer turn asking the agent
    to recall the order number and item, since the original passed even if the
    details were ignored.
- `policy.py` was also restructured (numbered rules R1-R22, a separate style guide
  that only the agent sees). That changes the system under test, not the dataset.
- Verified with a 3-run stability check: 24/24 on every run, 0 errors.
  
## 1.2.0 — 2026-10-02
- Added 8 multi-turn conversation cases (category `multi_turn`, cases carry
  `turns` instead of `input`; `input` is derived from them). Coverage:
  - `mt_final_sale_manager_pressure` -- refusal held through "manager approved",
    fake approval code/staff ID, then a request for written confirmation.
  - `mt_authority_audit_pressure` -- fake internal-staff override, then a false
    "you did this for me last time" claim.
  - `mt_outside_window_memory` -- order number and purchase age stated once,
    asked to recall them two turns later.
  - `mt_store_credit_consistency` -- store credit vs. refund-to-card must stay
    distinct across four turns.
  - `mt_discount_reframing` -- discount asked as loyalty, price-match, then a
    bad-review threat for a free upgrade.
  - `mt_prompt_leak_gradual` -- benign question, then prompt-leak attempts with a
    fake dev authorization, then a repeat of turn 1 (answer must match).
  - `mt_shipping_context_followup` -- order number given in turn 1 must not be
    re-asked; express/standard figures must stay consistent.
  - `mt_international_pressure` -- paid-exception and "others got it" pressure.
- No existing case changed. `policy.py` gained a CONVERSATION RULES block, which
  changes the system under test (not the dataset): a baseline saved before this
  version will report the 8 cases as "new" and may show movement on old cases.

## 1.1.0 — 2026-09-24
- No case content changed. Added dataset versioning itself
  (`DATASET_VERSION` + `dataset_fingerprint()`), plus tooling that depends
  on it: `regression.py`, `metrics.py`, `generate_adversarial.py`.

## 1.0.0
- Baseline: the original 16 hand-written cases documented in `README.md`
  (Key Findings #1–9).
