# MergeMind qwen35 Rewriter Experiment Results

## Summary

This report compares the existing `qwen35_rewriter` chain against the experimental
`qwen35_rewriter_sweci_contract`, `qwen35_rewriter_sweci_triage`, strict revision-guard,
file-aware revision, and before-snapshot revision profiles.

Primary success criterion:

- SWE-CI: reduce `actual_iterations`, or at least improve `gap_sequence`, `final_gap`, and `best_gap` on identical tasks.
- Real PR/MR review: produce grounded, useful comments on public merged PRs with reproducible metrics and artifacts.

## Agent Changes

- Baseline chain: `LLMGenerator -> LLMReranker -> LLMRewriter`.
- Contract chain: `SWEContractLLMGenerator -> SWEContractLLMReranker -> SWEContractLLMRewriter`.
- Triage chain: `SWETriageLLMGenerator -> SWETriageLLMReranker -> SWETriageLLMRewriter`.
- The new chains preserve the same number of agents, but change prompt intent:
  - generator acts as a correctness-only patch-risk reviewer;
  - reranker scores by likely repair impact;
  - rewriter emits an agent-facing revision contract.
- Strict revision guard keeps the same reviewer chain, but rejects programmer revisions that modify files outside the immediately preceding programmer patch.
- File-aware revision passes `/app/mergemind_review.md` and `/app/mergemind_allowed_files.txt` into the direct OpenAI programmer context. This matters because `direct_openai` is not a tool-using shell agent.
- Before-snapshot revision adds `/app/mergemind_before_files.md`, containing only the pre-programmer contents of files that the programmer already changed in the current epoch. This is not target/oracle information; it is the current epoch's before-state.
- `LLMJudge` remains optional and is used only for evaluation.

## SWE-CI Commands

Baseline template:

```bash
python scripts/run_swe_ci.py \
  --swe-ci-repo-path ~/SWE-CI \
  --tasks-path artifacts/swe_ci/tasks_lite.jsonl \
  --output-dir artifacts/swe_ci_runs \
  --run-id <baseline-run-id> \
  --limit 3 \
  --max-iterations 3 \
  --mode baseline \
  --splitting lite \
  --agent-name direct_openai \
  --base-url http://127.0.0.1:1234/v1 \
  --model-name qwen3.6-27b@iq2_xxs \
  --api-key lm-studio \
  --source-data-root ~/SWE-CI/data
```

Assisted template:

```bash
python scripts/run_swe_ci.py \
  --swe-ci-repo-path ~/SWE-CI \
  --tasks-path artifacts/swe_ci/tasks_lite.jsonl \
  --output-dir artifacts/swe_ci_runs \
  --run-id <assisted-run-id> \
  --limit 3 \
  --max-iterations 3 \
  --mode mergemind_assisted \
  --splitting lite \
  --agent-name direct_openai \
  --base-url http://127.0.0.1:1234/v1 \
  --model-name qwen3.6-27b@iq2_xxs \
  --api-key lm-studio \
  --source-data-root ~/SWE-CI/data \
  --mergemind-llm-provider local_qwen36_27b_iq2 \
  --mergemind-pipeline qwen35_rewriter_sweci_contract \
  --mergemind-top-n 3
```

Guarded triage template:

```bash
python scripts/run_swe_ci.py \
  --swe-ci-repo-path ~/SWE-CI \
  --tasks-path <single-task-or-lite-manifest>.jsonl \
  --output-dir artifacts/swe_ci_runs \
  --run-id <assisted-run-id> \
  --limit 1 \
  --max-iterations 3 \
  --mode mergemind_assisted \
  --splitting lite \
  --agent-name direct_openai \
  --base-url http://127.0.0.1:1234/v1 \
  --model-name qwen3.6-27b@iq2_xxs \
  --api-key lm-studio \
  --source-data-root ~/SWE-CI/data \
  --mergemind-llm-provider local_qwen36_27b_iq2 \
  --mergemind-pipeline qwen35_rewriter_sweci_triage \
  --mergemind-top-n 1 \
  --mergemind-min-score 0.75 \
  --mergemind-max-revision-epochs 2
```

## Real PR Commands

```bash
python scripts/run_github_pr_experiment.py \
  --manifest configs/pr_experiments/curated_real_prs.json \
  --config configs/base.yaml
```

Dry run:

```bash
python scripts/run_github_pr_experiment.py \
  --manifest configs/pr_experiments/curated_real_prs.json \
  --config configs/base.yaml \
  --dry-run
```

## Results

### SWE-CI

#### inline-snapshot task

Completed smoke artifacts on the Linux server:

- Baseline: `artifacts/swe_ci_runs/sweci_direct_lite_max3_001/summary.md`
- Contract assisted: `artifacts/swe_ci_runs/sweci_contract_lite_max3_001/summary.md`
- Comparison: `artifacts/swe_ci_runs/sweci_contract_ab_compare.md`

Task:

- `15r10nk__inline-snapshot__3bb05d__e2b9b2`
- `max_iterations=3`
- SWE-CI agent: `direct_openai`
- LM Studio model: `qwen3.6-27b@iq2_xxs`
- MergeMind provider: `local_qwen36_27b_iq2`
- MergeMind pipeline: `qwen35_rewriter_sweci_contract`

| run | gap sequence | actual iterations | final gap | best gap | duration sec | assisted reviews | assisted comments | assisted revisions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline `sweci_direct_lite_max3_001` | `16 -> 106 -> 13 -> 12` | 3 | 12 | 12 | 1314.574 | 0 | 0 | 0 |
| assisted contract `sweci_contract_lite_max3_001` | `16 -> 13 -> 12 -> 13` | 3 | 13 | 12 | 2025.850 | 3 | 9 | 3 |

Comparison result:

- Mean iteration delta: `0.000`
- Mean final gap delta: `+1.000`
- Assisted comments: `9`
- Assisted revisions: `3`

Interpretation:

- The run does not prove reduced `actual_iterations`; both paths consumed the configured 3 iterations and did not reach gap 0.
- The contract profile did avoid the baseline first-epoch regression: baseline went `16 -> 106`, while assisted went `16 -> 13`.
- The contract profile reached best gap `12` by epoch 2, while baseline reached `12` by epoch 3.
- Final gap was one point worse for assisted (`13` vs `12`), so the strict SWE-CI conclusion is mixed rather than better.
- No MergeMind review artifact used `target_sha`; each assist result records `target_sha_used_for_review=false`.

Per-epoch MergeMind usage for assisted contract:

| epoch | comments | LLM calls | tokens | review latency sec | parse error rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 3 | 8875 | 57.900 | 0.000 |
| 2 | 3 | 3 | 6925 | 49.591 | 0.000 |
| 3 | 3 | 3 | 7666 | 113.166 | 0.000 |

Triage follow-up on the same task:

- Run: `artifacts/swe_ci_runs/sweci_triage_top1_min075_max2_lite_max3_001/summary.md`
- Comparison: `artifacts/swe_ci_runs/sweci_triage_top1_ab_compare.md`
- Pipeline: `qwen35_rewriter_sweci_triage`
- Settings: `top_n=1`, `min_score=0.75`, `max_revision_epochs=2`

| run | gap sequence | actual iterations | final gap | best gap | assisted comments | assisted revisions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline `sweci_direct_lite_max3_001` | `16 -> 106 -> 13 -> 12` | 3 | 12 | 12 | 0 | 0 |
| triage `sweci_triage_top1_min075_max2_lite_max3_001` | `16 -> 13 -> 13 -> 12` | 3 | 12 | 12 | 2 | 2 |

Comparison result:

- Mean iteration delta: `0.000`
- Mean final gap delta: `0.000`
- Triage was better than the earlier contract profile on final gap (`12` vs `13`), but it did not beat baseline.

#### cle-b/httpdbg task

To avoid overfitting to the original `inline-snapshot` task, a second small-gap SWE-CI task was downloaded from the public dataset:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="skylenage-ai/SWE-CI",
    repo_type="dataset",
    allow_patterns=["data/cle-b__httpdbg__22e489__af88c4/**"],
    local_dir="/home/pashab/SWE-CI",
)
PY
```

Manifest:

- `artifacts/swe_ci/tasks_cle_b_httpdbg.jsonl`

Runs:

- Baseline: `artifacts/swe_ci_runs/sweci_direct_cle_b_max3_001/summary.md`
- Triage: `artifacts/swe_ci_runs/sweci_triage_cle_b_top1_min075_max2_max3_001/summary.md`
- Strict revision guard: `artifacts/swe_ci_runs/sweci_strict_cle_b_top1_min075_max2_max3_001/summary.md`
- Strict comparison: `artifacts/swe_ci_runs/sweci_strict_cle_b_ab_compare.md`
- File-aware revision: `artifacts/swe_ci_runs/sweci_fileaware_v2_cle_b_top1_min075_max2_max3_001/summary.md`
- Before-snapshot revision: `artifacts/swe_ci_runs/sweci_before_snapshot_cle_b_top1_min075_max2_max3_001/summary.md`
- Before-snapshot comparison: `artifacts/swe_ci_runs/sweci_before_snapshot_cle_b_ab_compare.md`

| run | gap sequence | actual iterations | final gap | best gap | duration sec | assisted comments | assisted revisions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline `sweci_direct_cle_b_max3_001` | `5 -> -1 -> 5 -> 11` | 3 | 11 | 5 | 413.506 | 0 | 0 |
| triage `sweci_triage_cle_b_top1_min075_max2_max3_001` | `5 -> -1 -> -1 -> 11` | 3 | 11 | 5 | 823.125 | 3 | 2 |
| strict guard `sweci_strict_cle_b_top1_min075_max2_max3_001` | `5 -> -1 -> -1 -> 5` | 3 | 5 | 5 | 620.393 | 3 | 2 |
| file-aware revision `sweci_fileaware_v2_cle_b_top1_min075_max2_max3_001` | `5 -> -1 -> 5 -> 5` | 3 | 5 | 5 | 715.597 | 3 | 2 |
| before-snapshot revision `sweci_before_snapshot_cle_b_top1_min075_max2_max3_001` | `5 -> 5 -> 5 -> 5` | 3 | 5 | 5 | 641.947 | 3 | 2 |

Best comparison against baseline:

- Mean iteration delta: `0.000`
- Mean final gap delta: `-6.000`
- Assisted comments: `3`
- Assisted revisions: `2`

Interpretation:

- This task still does not prove reduced `actual_iterations`.
- The strict revision guard does provide a positive SWE-CI artifact: it prevented the final regression from `5` to `11`, ending at final gap `5`.
- The guard caught a real bad revision attempt in epoch 2:
  - `revision_error=ValueError('MergeMind revision changed files outside the programmer patch: httpdbg/hooks/generic.py')`
  - The retry stayed inside `httpdbg/records.py`.
- The file-aware revision experiment exposed a direct-agent integration bug: MergeMind review text was being written into the container, but `direct_openai` did not include it in the model prompt context. After adding it, the revision pass consumed the comments, but epoch 1 still regressed to gap `-1` because the model was asked to restore a deleted file without seeing its previous contents.
- The before-snapshot revision experiment fixed that second issue by passing only the changed files' before-patch contents. It removed the epoch-1 regression (`5 -> -1` became `5 -> 5`) while preserving the same final gap improvement over baseline (`11 -> 5`).
- The ordinary triage profile generated plausible comments but did not improve the SWE-CI metric and cost more time than baseline.

Before-snapshot command:

```bash
SWE_CI_DIRECT_CONTEXT_CHARS=80000 SWE_CI_DIRECT_PROGRAMMER_MAX_TOKENS=9000 \
python scripts/run_swe_ci.py \
  --swe-ci-repo-path ~/SWE-CI \
  --tasks-path artifacts/swe_ci/tasks_cle_b_httpdbg.jsonl \
  --output-dir artifacts/swe_ci_runs \
  --run-id sweci_before_snapshot_cle_b_top1_min075_max2_max3_001 \
  --limit 1 \
  --max-iterations 3 \
  --timeout-seconds 10800 \
  --mode mergemind_assisted \
  --splitting lite \
  --agent-name direct_openai \
  --base-url http://127.0.0.1:1234/v1 \
  --model-name qwen3.6-27b@iq2_xxs \
  --api-key lm-studio \
  --source-data-root ~/SWE-CI/data \
  --mergemind-llm-provider local_qwen36_27b_iq2 \
  --mergemind-pipeline qwen35_rewriter_sweci_triage \
  --mergemind-top-n 1 \
  --mergemind-min-score 0.75 \
  --mergemind-max-revision-epochs 2
```

Important guardrail:

- All assist artifacts for these runs record `target_sha_used_for_review=false`.
- `/app/mergemind_before_files.md` contains only the source file contents from the current epoch before the programmer patch. It does not include target code, target diff, or hidden SWE-CI scoring information.

### Real PR Review

Completed artifact:

- `artifacts/github_pr_experiments/curated_real_prs_qwen35_contract_v2/summary.md`

Run settings:

- Manifest: `configs/pr_experiments/curated_real_prs.json`
- PRs: 5 public merged GitHub PRs
- Pipelines:
  - `qwen35_rewriter`
  - `qwen35_review_contract` aliasing `qwen35_rewriter_sweci_contract`
- Judge: enabled
- `limit_comments=3`
- `max_repository_files=5`
- Output token limits were not raised beyond the documented contract defaults: generator 1200, reranker 900, rewriter 1200.

Aggregate results:

| pipeline | PRs | ok | comments | gold PRs | hit@k | avg judge | avg grounded | avg useful | tokens | latency sec | fallbacks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen35_rewriter` | 5 | 5 | 12 | 5 | 0.000 | 0.650 | 0.730 | 0.660 | 57032 | 322.971 | 1 |
| `qwen35_rewriter_sweci_contract` | 5 | 5 | 15 | 5 | 0.000 | 0.794 | 0.920 | 0.790 | 76710 | 388.197 | 0 |

Real PR examples:

| PR | contract comment example | usefulness note |
| --- | --- | --- |
| [Vaquum/Limen#544](https://github.com/Vaquum/Limen/pull/544) | Reusing a `UEL` instance without clearing artifact lists can persist stale run data. | Useful patch-risk comment; directly grounded in state reset changes. |
| [truong0812/CrewAI_Review_Bot#17](https://github.com/truong0812/CrewAI_Review_Bot/pull/17) | Removing `_run_with_timeout` risks indefinite hangs if the new engine lacks equivalent timeout logic. | Useful behavioral compatibility comment; not text-similar to human review but actionable. |
| [aidilsaputrakirsan-classroom/cc-kelompok-a-hexacore#29](https://github.com/aidilsaputrakirsan-classroom/cc-kelompok-a-hexacore/pull/29) | Chaining `.filter()` after `.scalar()` raises at runtime; move `.filter(Fine.amount > 0)` before `.scalar()`. | Strong real bug-finding comment and highly actionable. |
| [aidilsaputrakirsan-classroom/cc-kelompok-a-hexacore#27](https://github.com/aidilsaputrakirsan-classroom/cc-kelompok-a-hexacore/pull/27) | Checking 503/504 before 401 can mask session expiration handling. | Plausible behavioral risk; useful but needs maintainer validation. |
| [pipeshub-ai/pipeshub-ai#2399](https://github.com/pipeshub-ai/pipeshub-ai/pull/2399) | Unbounded concurrent API requests may overwhelm Jira rate limits. | Potentially useful, but lower judge score and more speculative than the other PRs. |

Human-review similarity:

- `hit@k` stayed `0.000` for both pipelines.
- This is not treated as failure by itself because many generated findings are risk-oriented and do not lexically match human review comments in these public PRs.
- Judge and groundedness were more useful signals for this small curated set.

## Conclusion

The strict iteration-reduction criterion is not met yet: completed SWE-CI smokes did not reduce `actual_iterations`.

There is now positive SWE-CI evidence short of iteration reduction:

- On `inline-snapshot`, triage matched baseline final/best gap while using only two revision passes.
- On `cle-b/httpdbg`, strict/file-aware/before-snapshot profiles improved final gap from `11` to `5`, preventing a baseline/triage final regression.
- The strict guard produced an auditable safety signal by rejecting an out-of-patch revision attempt.
- The before-snapshot branch produced the cleanest `cle-b/httpdbg` trajectory: `5 -> 5 -> 5 -> 5`, avoiding both the baseline final regression and the earlier assisted epoch-1 pytest failure.

The stronger artifact remains the real-PR batch: across 5 public merged PRs, the contract profile improved judge score, groundedness, usefulness, produced more comments, and eliminated the fallback seen in the current `qwen35_rewriter` profile.

Recommended next experiment:

- Keep the strict revision guard, file-aware context, and before-snapshot context enabled for all in-loop SWE-CI tests.
- Add a real-source-file guard that skips MergeMind revision when the programmer patch only creates placeholder paths such as `src/package/module.py`.
- Run at least 3 small-gap SWE-CI tasks with `max_iterations=5`.
- Consider stopping or reducing comments after the gap stops improving, because epoch 3 regressed from best gap `12` to final gap `13`.
