# MergeMind Caveman SWE-CI Grid Results

## Summary

Branch: `codex/mergemind-caveman-sweci-grid`.

Primary metric is SWE-CI behavior, not standalone PR-review quality. The smoke grid used the fixed low-gap task `cle-b__httpdbg__22e489__af88c4` with identical baseline and assisted settings:

- SWE-CI agent: `direct_openai`
- model: `qwen3.6-27b@iq2_xxs`
- provider: local LM Studio through the Linux reverse tunnel
- `max_iterations=3`
- Docker network: `host`
- no `target_sha` in MergeMind review prompts or artifacts

The full five-task manifest is committed as `configs/swe_ci_caveman_low_gap_tasks.jsonl`, but this report only claims the completed one-task smoke. The `top2` run was stopped by the early-stop rule after 10+ minutes without reaching the first SWE-CI iteration.

## Integration Shape

MergeMind is integrated as an extra review context inside the SWE-CI loop, not as a post-run analyzer:

1. SWE-CI architect writes the requirement for the current epoch.
2. SWE-CI programmer makes the first code change.
3. MergeMind receives only the observed before/after diff for changed source files.
4. MergeMind generates a compact review comment.
5. The comment is passed back to the programmer as revision context.
6. The programmer performs a revision pass.
7. Pytest runs after the revision.

The assisted path does not pass `target_sha`, hidden target diff, or hidden oracle data into MergeMind. The `test_triage` variant additionally passes visible previous failed-test information, but only from already observed test output.

## Implemented Configs

New pipeline aliases:

- `qwen35_caveman_top1`: 3 agents, strict correctness-only, max 1 review comment per revision pass.
- `qwen35_caveman_top2`: 3 agents, max 2 comments for more recall.
- `qwen35_caveman_direct_top1`: generator emits the repair contract directly, reranker filters, no rewriter.
- `qwen35_caveman_test_triage`: top1-style chain plus previous visible failed-test summary.
- `qwen35_rewriter_sweci_safe_triage`: conservative triage after the `igrek51/wat` holdout; prefers no comment over broad repair, rejects generated/data/test/doc advice, and asks revisions to preserve currently passing behavior.
- Control: `qwen35_rewriter_sweci_triage`.

Pipeline-specific token limits are configured through `llm_pipeline_overrides` in `configs/base.yaml`, so conservative output budgets are scoped to the selected pipeline instead of changing global defaults.

## Commands

Baseline:

```bash
python scripts/run_swe_ci.py \
  --swe-ci-repo-path /home/pashab/SWE-CI \
  --tasks-path configs/swe_ci_caveman_low_gap_tasks.jsonl \
  --output-dir artifacts/swe_ci_runs \
  --run-id caveman_grid_baseline_001 \
  --limit 1 \
  --max-iterations 3 \
  --timeout-seconds 10800 \
  --mode baseline \
  --splitting lite \
  --api-key lm-studio \
  --base-url http://127.0.0.1:1234/v1 \
  --model-name qwen3.6-27b@iq2_xxs \
  --agent-name direct_openai \
  --source-data-root /home/pashab/SWE-CI/data \
  --docker-network host
```

Assisted template:

```bash
python scripts/run_swe_ci.py \
  --swe-ci-repo-path /home/pashab/SWE-CI \
  --tasks-path configs/swe_ci_caveman_low_gap_tasks.jsonl \
  --output-dir artifacts/swe_ci_runs \
  --run-id <run-id> \
  --limit 1 \
  --max-iterations 3 \
  --timeout-seconds 10800 \
  --mode mergemind_assisted \
  --splitting lite \
  --api-key lm-studio \
  --base-url http://127.0.0.1:1234/v1 \
  --model-name qwen3.6-27b@iq2_xxs \
  --agent-name direct_openai \
  --source-data-root /home/pashab/SWE-CI/data \
  --docker-network host \
  --mergemind-config configs/base.yaml \
  --mergemind-pipeline <pipeline> \
  --mergemind-llm-provider local_qwen36_27b_iq2 \
  --mergemind-top-n <1-or-2> \
  --mergemind-min-score <0.75-or-0.70> \
  --mergemind-max-revision-epochs 2
```

Comparison:

```bash
python scripts/summarize_swe_ci_grid.py \
  --baseline-run-dir artifacts/swe_ci_runs/caveman_grid_baseline_001 \
  --assisted-run-dirs \
    artifacts/swe_ci_runs/caveman_grid_control_triage_cle_b_001 \
    artifacts/swe_ci_runs/caveman_grid_top1_cle_b_001 \
    artifacts/swe_ci_runs/caveman_grid_direct_top1_cle_b_001 \
    artifacts/swe_ci_runs/caveman_grid_test_triage_cle_b_001 \
  --output-dir artifacts/swe_ci_runs/caveman_grid_smoke_summary_001
```

The comparison pipeline now also parses the official SWE-CI stdout table and carries
`EVOSCORE(GAMMA=1)`, `SOLVED_RATE`, and `ZERO_REGRESSION` into `summary.json` and
the generated markdown reports. This keeps the custom gap/failing-test metrics
aligned with the benchmark's own scoring output.

## Results

Server artifact root:

- `/home/pashab/MergeMind-caveman-grid/artifacts/swe_ci_runs/caveman_grid_smoke_summary_001/summary.md`

Baseline:

- Run: `caveman_grid_baseline_001`
- Gap sequence: `5 -> 11 -> -1 -> 11`
- Final gap: `11`
- Best gap: `5`
- Tokens: `67006`

Grid summary:

| config | official evoscore | gap delta | final invalid | invalid iters | failed-set jaccard | fixed | new | comments | revisions | tokens | review tokens | LLM calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | `-0.5439` | n/a | 0 | 1 | n/a | n/a | n/a | 0 | 0 | 67006 | 0 | 0 |
| `qwen35_caveman_top1` | `-0.3333` | `-6` | 0 | 1 | 0.949 | 6 | 0 | 2 | 2 | 63205 | 7935 | 6 |
| `qwen35_caveman_direct_top1` | `-0.3333` | `-6` | 0 | 1 | 0.949 | 6 | 0 | 2 | 1 | 66863 | 8966 | 5 |
| `qwen35_rewriter_sweci_triage` | `0.0000` | `-6` | 0 | 0 | 0.949 | 6 | 0 | 1 | 1 | 69538 | 14405 | 9 |
| `qwen35_caveman_test_triage` | `-0.3333` | n/a | 1 | 1 | 0.000 | n/a | n/a | 2 | 1 | 59803 | 12548 | 7 |

The `test_triage` row is not considered a valid winner because its final gap was `-1`. The comparator now excludes invalid final gaps from improvement averages and marks them explicitly.

The `top2` run was started as `caveman_grid_top2_cle_b_001` but stopped before completion. It stayed in the first SWE-CI generation phase for 10+ minutes, produced no `iteration.jsonl`, and created no MergeMind assist artifacts. It is treated as an early-stop/no-progress result, not as a scored run.

## Metric Notes

`actual_iterations` did not improve in the completed runs. Baseline and completed assisted variants all used the configured 3 iterations.

`final_gap` is the number of failing tests at the last recorded SWE-CI iteration. Lower is better, but equal numeric gaps do not necessarily mean the same tests are failing.

For that reason the comparison also records failed-test-set metrics:

- `failed-set jaccard`: overlap between the final baseline failing tests and final assisted failing tests.
- `fixed`: baseline final failures that are absent in the assisted final set.
- `new`: assisted final failures that were not in the baseline final set.

Invalid gap values such as `-1` are not treated as improvements. They are counted separately through `final invalid` and `invalid iters`.

The official SWE-CI stdout `EVOSCORE(GAMMA=1)` gives a slightly different ranking than the custom grid summary:

- best official evoscore in this smoke: `qwen35_rewriter_sweci_triage` with `0.0000`;
- best final-gap/token tradeoff in this smoke: `qwen35_caveman_top1`, because it matches the final gap improvement while using fewer total and review tokens.

## Token Budget

Measured completed smoke usage:

- Baseline coding tokens: `67006`
- Assisted total tokens across scored configs: `259409`
- Assisted MergeMind review tokens across scored configs: `43854`
- Total measured completed smoke tokens: `326415`

The planned aggressive 5-task grid budget was `2.8M-3.6M` tokens with a hard review ceiling of `4.0M`. This smoke used about `0.33M` measured tokens before stopping the slow `top2` variant. The aborted `top2` first-generation call is not included because it never wrote SWE-CI iteration accounting.

## Conclusion

`qwen35_caveman_top1` is the current winner for this smoke:

- It improved final gap from `11` to `5`.
- It fixed 6 baseline final failures with no new final failures.
- It used fewer measured total tokens than baseline: `63205` vs `67006`.
- It used fewer tokens and fewer review tokens than the old triage control while achieving the same valid final gap improvement.

If the primary criterion is the official SWE-CI evoscore, the current best scored variant is still the control `qwen35_rewriter_sweci_triage`, not caveman top1.

This still does not prove reduced `actual_iterations`: baseline and all completed assisted runs used the configured 3 iterations. The evidence is a smaller final gap, better failed-test-set outcome, and lower measured token usage for `qwen35_caveman_top1` on the completed smoke task.

## Holdout Follow-up: igrek51/wat

Date: 2026-05-27.

Task:

- `igrek51__wat__ecddda__8efafa`
- initial gap: `5`
- model: `qwen3.6-27b@iq2_xxs`
- `max_iterations=5`
- server artifact root: `/home/pashab/MergeMind-caveman-grid/artifacts/swe_ci_runs`

Runs:

| run | status | gap sequence | final gap | best gap | comments | total tokens | notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `holdout_baseline_igrek51_max5_002` | completed | `5 -> -1 -> -1 -> 5 -> 5 -> 5` | 5 | 5 | 0 | 91318 | baseline did not improve the initial gap; two invalid pytest epochs |
| `holdout_caveman_top1_igrek51_max5_001` | early-stopped | `5 -> 40 -> 25` | 25 | 5 | 2 | 94270 | worse than baseline; first MergeMind revision increased failures from 5 to 40 |
| `holdout_triage_igrek51_max5_001` | early-stopped | `5 -> 5` | 5 | 5 | 1 | 22382 | no gap improvement before stop; epoch 2 revision hit malformed JSON after a long generation |
| `holdout_safe_triage_igrek51_max5_001` | early-stopped | `5 -> -1 -> -1` | -1 | 5 | 2 | 54104 | no useful progress; two invalid pytest epochs, first revision also hit the outside-patch guard |

The `qwen35_caveman_top1` comment was plausible at text level:

> Implement `.public` as a modifier that filters private attributes.

However, the revision pass broadened the code change and regressed the run to 40
failing tests. This is a negative result for the current caveman prompt on this
holdout task. The result suggests that the prompt should make the revision
contract stricter: prefer preserving existing behavior over implementing a
larger feature-shaped repair when confidence comes only from a diff/requirement
match.

The control `qwen35_rewriter_sweci_triage` was safer on the first completed
epoch: it kept the gap at 5 instead of worsening it, but it did not improve the
task before the run was stopped. Its second revision attempt produced malformed
JSON, which points to a separate reliability issue in the direct local-model
agent output path.

Current conclusion for this holdout:

- no reduction in `actual_iterations` was shown;
- no `final_gap` improvement was shown;
- `qwen35_caveman_top1` is not a good default for this task;
- `qwen35_rewriter_sweci_triage` remains safer than caveman on this task, but not yet useful;
- `qwen35_rewriter_sweci_safe_triage` did not help on this task and was stopped to save GPU time after two invalid epochs.

Follow-up implementation from this negative holdout:

- the direct local-model agent no longer uses `src/package/module.py` as an example path in its JSON prompt, because Qwen sometimes copied that placeholder literally;
- during MergeMind revision passes, the direct agent now enforces `/app/mergemind_allowed_files.txt` before writing file replacements;
- an outside-patch revision is now treated as a non-retryable guard failure, so the same forbidden edit is not retried 10 times.

Next prompt/config direction:

- reject comments when the programmer patch touches unrelated/generated-looking files;
- require the reviewer to prefer `0 comments` when the diff does not clearly touch the failing behavior;
- keep the revision contract focused on preserving the current passing tests.

Executed safe-triage command:

```bash
python scripts/run_swe_ci.py \
  --swe-ci-repo-path /home/pashab/SWE-CI \
  --tasks-path configs/swe_ci_caveman_holdout_tasks.jsonl \
  --output-dir artifacts/swe_ci_runs \
  --run-id holdout_safe_triage_igrek51_max5_001 \
  --limit 1 \
  --max-iterations 5 \
  --timeout-seconds 14400 \
  --mode mergemind_assisted \
  --splitting lite \
  --api-key lm-studio \
  --base-url http://127.0.0.1:1234/v1 \
  --model-name qwen3.6-27b@iq2_xxs \
  --agent-name direct_openai \
  --source-data-root /home/pashab/SWE-CI/data \
  --docker-network host \
  --mergemind-config configs/base.yaml \
  --mergemind-pipeline qwen35_rewriter_sweci_safe_triage \
  --mergemind-llm-provider local_qwen36_27b_iq2 \
  --mergemind-top-n 1 \
  --mergemind-min-score 0.80 \
  --mergemind-max-revision-epochs 1
```

## Short Status Text

Current concise status for reporting:

> MergeMind was integrated as an additional review context inside the SWE-CI loop. The agent first makes a code change, MergeMind reviews the diff and returns a comment, the agent performs a revision pass using that comment, and tests run only after that revision. The local model was Qwen3.6-27B GGUF IQ2_XXS through LM Studio with 32768 context, exposed to the Ubuntu server via reverse SSH tunnel. The result is preliminary: iteration count has not decreased yet, but on `cle-b/httpdbg` baseline finished with final gap 11 and the best MergeMind variants finished with final gap 5. The official SWE-CI evoscore improved from `-0.5439` baseline to `0.0000` for the current triage-control assisted run.
