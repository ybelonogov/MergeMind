# SWE-CI Batch For NIR

Date: 2026-05-30.

Server worktree:

- `/home/pashab/MergeMind-caveman-grid`

Model:

- LM Studio OpenAI-compatible API
- `base_url=http://127.0.0.1:1234/v1`
- `model_name=qwen3.6-27b@iq2_xxs`
- context length in LM Studio: `32768`

SWE-CI agent:

- `direct_openai`

## Goal

Run more than a one-task smoke so the NIR write-up can separate anecdotal results from a small reproducible batch.

The comparison uses the same fixed low-gap SWE-CI task order:

1. `cle-b__httpdbg__22e489__af88c4`
2. `igrek51__wat__ecddda__8efafa`
3. `kinnay__nintendoclients__e2b673__72846c`
4. `eliben__pycparser__7f6b34__6ba954`

The fifth task from `configs/swe_ci_caveman_low_gap_tasks.jsonl`,
`gusye1234__nano-graphrag__bc175d__9b33a7`, did not create a SWE-CI result file in the baseline run, so it is excluded from the clean A/B comparison.

## Infrastructure Fix Before Batch

The first baseline attempt was interrupted because `direct_openai` rejected model outputs that used safe container paths such as:

```text
/app/httpdbg/records.py
```

The adapter previously accepted `/app/code/...` and `code/...`, but not `/app/...`. This caused repeated programmer retries and wasted GPU time. The adapter now normalizes safe in-container paths to repo-relative paths before applying the existing traversal/test-file guards.

Local verification:

```bash
python -m unittest \
  tests.test_direct_openai_agent_template \
  tests.test_swe_ci_assisted_workdir \
  tests.test_swe_ci_assist_helper \
  tests.test_llm \
  tests.test_pipeline_modes \
  tests.test_swe_ci_result_parser \
  tests.test_compare_swe_ci_runs
```

Result:

```text
Ran 55 tests
OK
```

## Commands

Baseline:

```bash
python scripts/run_swe_ci.py \
  --swe-ci-repo-path /home/pashab/SWE-CI \
  --tasks-path configs/swe_ci_caveman_low_gap_tasks.jsonl \
  --output-dir artifacts/swe_ci_runs \
  --run-id nir5_baseline_max3_002 \
  --limit 5 \
  --max-iterations 3 \
  --timeout-seconds 7200 \
  --mode baseline \
  --splitting lite \
  --agent-name direct_openai \
  --base-url http://127.0.0.1:1234/v1 \
  --model-name qwen3.6-27b@iq2_xxs \
  --api-key lm-studio \
  --source-data-root /home/pashab/SWE-CI/data \
  --docker-network host
```

Assisted:

```bash
python scripts/run_swe_ci.py \
  --swe-ci-repo-path /home/pashab/SWE-CI \
  --tasks-path configs/swe_ci_caveman_low_gap_tasks.jsonl \
  --output-dir artifacts/swe_ci_runs \
  --run-id nir4_test_guard_max3_001 \
  --limit 4 \
  --max-iterations 3 \
  --timeout-seconds 7200 \
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

Comparison:

```bash
python scripts/compare_swe_ci_runs.py \
  --baseline-run-dir artifacts/swe_ci_runs/nir5_baseline_max3_002 \
  --assisted-run-dir artifacts/swe_ci_runs/nir4_test_guard_max3_001 \
  --output artifacts/swe_ci_runs/nir4_test_guard_comparison_001.md \
  --json-output artifacts/swe_ci_runs/nir4_test_guard_comparison_001.json
```

## Artifacts

- baseline summary: `/home/pashab/MergeMind-caveman-grid/artifacts/swe_ci_runs/nir5_baseline_max3_002/summary.md`
- assisted summary: `/home/pashab/MergeMind-caveman-grid/artifacts/swe_ci_runs/nir4_test_guard_max3_001/summary.md`
- comparison report: `/home/pashab/MergeMind-caveman-grid/artifacts/swe_ci_runs/nir4_test_guard_comparison_001.md`
- comparison JSON: `/home/pashab/MergeMind-caveman-grid/artifacts/swe_ci_runs/nir4_test_guard_comparison_001.json`

## Aggregate Result

| metric | baseline | MergeMind test_guard |
| --- | ---: | ---: |
| clean compared tasks | 4 | 4 |
| actual iterations avg | 3.000 | 3.000 |
| valid final gap count | 0 | 0 |
| invalid final gap count | 4 | 4 |
| average best gap | 6.250 | 6.250 |
| assisted comments | 0 | 7 |
| assisted revisions | 0 | 2 |
| total visible tokens | 353623 | 499411 |
| MergeMind review tokens | 0 | 135610 |
| MergeMind LLM calls | 0 | 32 |
| mean official EvoScore delta | n/a | -0.070 |

`final_gap=-1` means pytest did not execute correctly for the final iteration, so final numeric gap deltas are not meaningful for these rows. The valid comparison signal is therefore: best gap, invalid iteration count, failed-set overlap where available, comments/revisions, token use, and EvoScore delta.

## Per-Task Result

| task | baseline gaps | assisted gaps | baseline best | assisted best | assisted comments | assisted revisions | assisted review tokens | result |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cle-b/httpdbg` | `5 -> 5 -> -1 -> -1` | `5 -> -1 -> -1 -> -1` | 5 | 5 | 1 | 0 | 15130 | worse EvoScore; no score improvement |
| `igrek51/wat` | `5 -> 43 -> -1 -> -1` | `5 -> -1 -> -1 -> -1` | 5 | 5 | 2 | 0 | 30158 | avoided numeric `43` iteration, but final still invalid |
| `kinnay/NintendoClients` | `7 -> 27 -> -1 -> -1` | `7 -> 10 -> -1 -> -1` | 7 | 7 | 3 | 1 | 57534 | smaller first-regression gap, but final still invalid |
| `eliben/pycparser` | `8 -> -1 -> -1 -> -1` | `8 -> -1 -> -1 -> -1` | 8 | 8 | 1 | 1 | 32788 | no score improvement |

## Interpretation

This batch does not show that `qwen35_rewriter_sweci_test_guard` improves SWE-CI solving quality.

The main result is negative:

- no task reached a lower best gap than baseline;
- no task reduced `actual_iterations`;
- all final gaps were invalid in both baseline and assisted;
- assisted mode spent about `145788` more visible tokens than baseline on the four compared tasks;
- MergeMind used `135610` review tokens and 32 review LLM calls;
- the average official EvoScore delta was `-0.070`.

There are two limited positive signals:

- on `igrek51/wat`, assisted avoided the baseline's numeric `43` failed-test iteration, but still ended invalid;
- on `kinnay/NintendoClients`, assisted first regression was `10` instead of baseline `27`, but the revision did not prevent later invalid pytest.

For the NIR text this should be described as an important failed configuration, not as evidence of improvement. It supports the claim that comment generation must be evaluated inside the solving loop, because plausible comments can still fail to improve SWE-CI outcomes and can add significant token cost.

## Main Bottleneck Observed

The biggest practical blocker is no longer only MergeMind prompt quality. The `direct_openai` file-replacement protocol is fragile with the local Qwen model:

- the model often emits whole-file replacements for large files;
- long JSON responses are truncated or malformed;
- malformed JSON causes retry loops;
- retries consume GPU time and often still end in `final_gap=-1`.

For the next experimental branch, the strongest change is likely to replace full-file JSON replacement with a patch/diff protocol or a smaller edit contract before continuing prompt tuning.

## Conclusion

This run gives a reproducible four-task baseline for the NIR, but `qwen35_rewriter_sweci_test_guard` is not the winning MergeMind configuration.

Current defensible statement:

> On a four-project SWE-CI batch with local Qwen, the strict test-aware MergeMind reviewer did not improve final SWE-CI results and increased token usage. However, the batch exposed concrete failure modes of the current agent interface and gives a reproducible basis for the next iteration: patch-format edits, lower retry rate, and less conservative review filtering.
