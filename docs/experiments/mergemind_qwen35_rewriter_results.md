# MergeMind qwen35 Rewriter Experiment Results

## Summary

This report compares the existing `qwen35_rewriter` chain against the experimental `qwen35_rewriter_sweci_contract` profile.

Primary success criterion:

- SWE-CI: reduce `actual_iterations`, or at least improve `gap_sequence`, `final_gap`, and `best_gap` on identical tasks.
- Real PR/MR review: produce grounded, useful comments on public merged PRs with reproducible metrics and artifacts.

## Agent Changes

- Baseline chain: `LLMGenerator -> LLMReranker -> LLMRewriter`.
- New contract chain: `SWEContractLLMGenerator -> SWEContractLLMReranker -> SWEContractLLMRewriter`.
- The new chain preserves the same number of agents, but changes prompt intent:
  - generator acts as a correctness-only patch-risk reviewer;
  - reranker scores by likely repair impact;
  - rewriter emits an agent-facing revision contract.
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

The strict SWE-CI success criterion is not met on the completed one-task smoke: `actual_iterations` did not decrease and final gap was slightly worse for the contract profile.

There is still evidence that in-loop MergeMind helps the repair trajectory: the assisted run avoided the first-epoch regression and reached the baseline best gap one epoch earlier. The stronger artifact is the real-PR batch: across 5 public merged PRs, the contract profile improved judge score, groundedness, usefulness, produced more comments, and eliminated the fallback seen in the current `qwen35_rewriter` profile.

Recommended next experiment:

- Run the same A/B on at least 3 SWE-CI tasks with `max_iterations=5`.
- Add a guard that skips the revision pass when MergeMind comments are low-confidence or repetitive.
- Consider stopping or reducing comments after the gap stops improving, because epoch 3 regressed from best gap `12` to final gap `13`.
