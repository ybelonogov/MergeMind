# SWE-CI Pair30 Baseline vs MergeMind

This experiment compares the same SWE-CI tasks in two modes:

- `baseline`: SWE-CI with `direct_openai`, no MergeMind comments.
- `assisted`: SWE-CI with `direct_openai` plus MergeMind `qwen35_rewriter_sweci_triage`.

Both modes use `max_iterations=5`.

The repository stores the fixed task manifest, the runner scripts, and compact
experiment artifacts. Raw SWE-CI logs and prompt logs are intentionally not
committed here.

## Prepare Tasks

The fixed task list is committed in:

- `configs/swe_ci_nir_pair30_tasks.jsonl`

Regenerate chunk files before a run:

```bash
python scripts/prepare_swe_ci_pair_manifest.py \
  --source-tasks-path configs/swe_ci_nir_pair30_tasks.jsonl \
  --output-dir configs \
  --output-stem swe_ci_nir_pair30_tasks \
  --limit 30 \
  --chunk-size 5
```

This writes:

- `configs/swe_ci_nir_pair30_tasks.jsonl`
- `configs/swe_ci_nir_pair30_tasks_chunk_01.jsonl` ... `chunk_06.jsonl`
- `configs/swe_ci_nir_pair30_tasks_manifest_info.json`

## Smoke

Before the full run, verify one task with `max_iterations=1`:

```bash
python scripts/run_swe_ci_pair_chunks.py \
  --swe-ci-repo-path <swe_ci_root> \
  --chunks-dir configs \
  --chunk-glob swe_ci_nir_pair30_tasks_chunk_01.jsonl \
  --output-root artifacts/swe_ci_runs \
  --run-id nir_pair30_smoke \
  --source-data-root <swe_ci_data_root> \
  --max-iterations 1 \
  --timeout-seconds 7200
```

Check that:

- `paired_summary.md` and `paired_summary.json` are written.
- baseline task artifacts contain `prompt_logs/<task_id>/direct_openai/*.jsonl`.
- assisted epoch artifacts contain `mergemind_assist/<task_id>/epoch_*/prompt_logs/*.jsonl`.

## Pair30 Run

Keep the LM Studio reverse tunnel open on Windows:

```powershell
ssh -N -R 1234:127.0.0.1:1234 -p 7090 user@server
```

Run chunks on the Linux server:

```bash
python scripts/run_swe_ci_pair_chunks.py \
  --swe-ci-repo-path <swe_ci_root> \
  --chunks-dir configs \
  --chunk-glob 'swe_ci_nir_pair30_tasks_chunk_*.jsonl' \
  --output-root artifacts/swe_ci_runs \
  --run-id nir_pair30_qwen36_triage_max5 \
  --source-data-root <swe_ci_data_root> \
  --base-url http://127.0.0.1:1234/v1 \
  --model-name 'qwen3.6-27b@iq2_xxs' \
  --api-key lm-studio \
  --docker-network host \
  --max-iterations 5 \
  --timeout-seconds 7200 \
  --mergemind-pipeline qwen35_rewriter_sweci_triage \
  --mergemind-llm-provider local_qwen36_27b_iq2 \
  --mergemind-top-n 1 \
  --mergemind-min-score 0.75 \
  --mergemind-max-revision-epochs 5
```

The current committed artifact set is a partial pair30 run. It includes baseline
chunks 01--03 and assisted chunks 01--03. After combining the completed chunks,
the baseline side contains 15 task rows, the assisted side contains 11 task rows,
and the paired comparison contains the tasks that are present in both sides.

Committed compact artifacts:

- `docs/experiments/artifacts/pair30/swe_ci_runs/nir_pair30_smoke_cle_b_max1/`
- `docs/experiments/artifacts/pair30/swe_ci_runs/nir_pair30_qwen36_triage_max5/`
- `docs/experiments/artifacts/pair30/swe_ci_runs/nir_pair30_qwen36_triage_max5/paired_summary.md`
- `docs/experiments/artifacts/pair30/swe_ci_runs/nir_pair30_qwen36_triage_max5/paired_summary.json`

Current partial-run summary:

- baseline combined tasks: `15`
- assisted combined tasks: `11`
- compared successful task pairs: `10`
- mean iteration delta: `0.000`
- mean iterations-to-best-gap delta: `-0.200`
- mean official EvoScore delta: `-0.041`
- improved / worse / unchanged / incomplete: `1 / 5 / 5 / 4`
- assisted comments: `46`
- assisted revisions: `46`
- assisted review tokens: `379598`

These numbers describe only the partial run above. They must not be described as
a completed 30-task result.

## Metrics

The paired summary includes:

- gap trajectory for baseline and assisted;
- iterations to best gap;
- final gap and best gap;
- official solved-rate and EvoScore deltas;
- failed-test overlap, fixed failures, and new failures;
- MergeMind comment/revision counts;
- total tokens, review tokens, LLM calls, and duration;
- task-level label: `improved`, `worse`, `unchanged`, or `incomplete`.

`target_sha` is not passed to MergeMind prompts or review artifacts.
