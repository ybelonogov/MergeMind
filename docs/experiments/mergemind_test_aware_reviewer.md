# MergeMind Test-Aware Reviewer

Date: 2026-05-29.

Branch: `codex/mergemind-test-aware-reviewer`.

## Goal

The previous SWE-CI experiments showed two useful facts:

- MergeMind can improve the final failed-test gap on `cle-b/httpdbg`.
- MergeMind can also hurt or waste tokens when a plausible comment is not tied tightly enough to visible test failures.

This branch adds a stricter reviewer mode intended for SWE-CI, not for general human PR review.

## New Pipeline

Pipeline mode:

- `qwen35_rewriter_sweci_test_guard`

Alias:

- `qwen35_review_test_guard`

Design:

- include visible previous pytest failures in the MergeMind review context;
- build the reviewed diff from Python source files only;
- generate at most one comment;
- prefer `0 comments` unless there is a direct chain from failing test to changed Python code;
- rewrite the selected comment into an agent-facing repair contract.

Required contract fields:

```text
Failing test:
Finding:
Evidence:
Expected revision:
Do not change:
Confidence:
```

## Why This Is Different

Earlier modes were still too review-like:

- `qwen35_rewriter_sweci_triage` could produce useful root-cause comments, but it did not require a visible failed-test link.
- `qwen35_rewriter_sweci_safe_triage` was safer, but it could still spend tokens on comments that did not improve the gap.
- `qwen35_caveman_top1` improved `cle-b/httpdbg`, but worsened `igrek51/wat`.

The new mode treats a comment as useful only if it can plausibly reduce the next SWE-CI gap. If that evidence is missing, the correct output is no comment.

## Implementation Notes

Code changes:

- `src/models/llm.py`
  - `SWETestGuardLLMGenerator`
  - `SWETestGuardLLMReranker`
  - `SWETestGuardLLMRewriter`
- `src/inference/factory.py`
  - new pipeline mode and alias;
  - pipeline component wiring.
- `src/validation/swe_ci/assist_helper.py`
  - failure context is enabled for `test_guard`;
  - reviewed diff is restricted to `.py` files for `test_guard`.
- `src/validation/swe_ci/assisted.py`
  - MergeMind revision allow-list is restricted to Python source files from the programmer patch;
  - non-code files are recorded as skipped.

The mode still does not use `target_sha` or hidden solution data.

## Suggested Smoke Command

```bash
python scripts/run_swe_ci.py \
  --swe-ci-repo-path /home/pashab/SWE-CI \
  --tasks-path configs/swe_ci_caveman_holdout_tasks.jsonl \
  --output-dir artifacts/swe_ci_runs \
  --run-id test_guard_igrek51_max5_001 \
  --limit 1 \
  --max-iterations 5 \
  --timeout-seconds 14400 \
  --mode mergemind_assisted \
  --splitting lite \
  --agent-name direct_openai \
  --base-url http://127.0.0.1:1234/v1 \
  --model-name qwen3.6-27b@iq2_xxs \
  --api-key lm-studio \
  --source-data-root /home/pashab/SWE-CI/data \
  --docker-network host \
  --mergemind-config configs/base.yaml \
  --mergemind-pipeline qwen35_rewriter_sweci_test_guard \
  --mergemind-llm-provider local_qwen36_27b_iq2 \
  --mergemind-top-n 1 \
  --mergemind-min-score 0.80 \
  --mergemind-max-revision-epochs 1
```

## Smoke Attempt

Run id:

- `test_guard_igrek51_max3_001`

Task:

- `igrek51__wat__ecddda__8efafa`

Setting:

- `max_iterations=3`
- `model_name=qwen3.6-27b@iq2_xxs`
- `pipeline=qwen35_rewriter_sweci_test_guard`
- server artifact path: `/home/pashab/MergeMind-caveman-grid/artifacts/swe_ci_runs/test_guard_igrek51_max3_001`

Result:

- not scored;
- initial SWE-CI gap was recorded as `5`;
- the run did not reach the MergeMind review step;
- SWE-CI architect retries failed with LM Studio `HTTP 400`:

```text
Failed to load model "qwen3.6-27b@iq2_xxs". Error: Error loading model.
```

The run was stopped manually to avoid spending GPU time on repeated model-load failures. This is an infrastructure/model-loading failure, not evidence for or against the new reviewer pipeline.

## Acceptance Criteria

Treat this mode as better only if one of these holds:

- lower `actual_iterations` to the same or lower gap;
- lower `final_gap` or `best_gap`;
- fewer new failing tests at the same numeric gap;
- same result with fewer MergeMind comments, revision attempts, LLM calls, or tokens.

Do not count a run as improved if the gap is `-1`, if new unrelated failing tests appear, or if the result costs substantially more tokens without a SWE-CI metric improvement.
