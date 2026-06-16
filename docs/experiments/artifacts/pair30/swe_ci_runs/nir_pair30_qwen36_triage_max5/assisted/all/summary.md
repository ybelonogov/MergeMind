# SWE-CI Run Summary: assisted_all

- Generated at: 2026-06-16T13:58:01.443380+00:00
- Tasks: 11
- Success: 10
- Failed: 0
- Timeout: 1
- Skipped: 0
- Pass rate: 0.909
- Average duration seconds: 1839.248
- Average actual iterations: 5.000
- Average final gap: 62.400
- Invalid final gap count: 5
- Average best gap: 25.100
- Gap zero count: 0
- MergeMind reviewed tasks: 0
- MergeMind skipped reviews: 0
- MergeMind comments: 0
- MergeMind assisted reviews: 46
- MergeMind assisted comments: 46
- MergeMind assisted revisions: 46
- Total tokens: 2461631
- MergeMind review tokens: 379598
- MergeMind LLM calls: 139

## Tasks

| task_id | status | iterations | final_gap | best_gap | duration_sec | tokens | review_tokens | exit_code | mergemind_review | comments | assisted_comments | stdout | stderr | error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | --- |
| 15r10nk__inline-snapshot__3bb05d__e2b9b2 | success | 5 | 106 | 12 | 2259.742 | 257268 | 25078 | 0 |  | 0 | 4 | [stdout](<experiment_root>/artifacts/swe_ci_runs/nir_pair30_qwen36_triage_max5/assisted/chunk_01/logs/15r10nk__inline-snapshot__3bb05d__e2b9b2/stdout.log) | [stderr](<experiment_root>/artifacts/swe_ci_runs/nir_pair30_qwen36_triage_max5/assisted/chunk_01/logs/15r10nk__inline-snapshot__3bb05d__e2b9b2/stderr.log) |  |
| 9001__copyparty__452592__4bdcbc | success | 5 | 16 | 16 | 620.548 | 267706 | 43000 | 0 |  | 0 | 5 | [stdout](<experiment_root>/artifacts/swe_ci_runs/nir_pair30_qwen36_triage_max5/assisted/chunk_01/logs/9001__copyparty__452592__4bdcbc/stdout.log) | [stderr](<experiment_root>/artifacts/swe_ci_runs/nir_pair30_qwen36_triage_max5/assisted/chunk_01/logs/9001__copyparty__452592__4bdcbc/stderr.log) |  |
| aio-libs__yarl__576b5e__857930 | success | 5 | -1 | 57 | 790.518 | 251332 | 35648 | 0 |  | 0 | 5 | [stdout](<experiment_root>/artifacts/swe_ci_runs/nir_pair30_qwen36_triage_max5/assisted/chunk_01/logs/aio-libs__yarl__576b5e__857930/stdout.log) | [stderr](<experiment_root>/artifacts/swe_ci_runs/nir_pair30_qwen36_triage_max5/assisted/chunk_01/logs/aio-libs__yarl__576b5e__857930/stderr.log) |  |
| amaranth-lang__amaranth__46e0a0__a4402b | success | 5 | -1 | 37 | 1299.429 | 285181 | 45681 | 0 |  | 0 | 5 | [stdout](<experiment_root>/artifacts/swe_ci_runs/nir_pair30_qwen36_triage_max5/assisted/chunk_01/logs/amaranth-lang__amaranth__46e0a0__a4402b/stdout.log) | [stderr](<experiment_root>/artifacts/swe_ci_runs/nir_pair30_qwen36_triage_max5/assisted/chunk_01/logs/amaranth-lang__amaranth__46e0a0__a4402b/stderr.log) |  |
| anthropics__claude-agent-sdk-python__91315e__22fa9f | success | 5 | 40 | 24 | 1230.574 | 212729 | 10479 | 0 |  | 0 | 2 | [stdout](<experiment_root>/artifacts/swe_ci_runs/nir_pair30_qwen36_triage_max5/assisted/chunk_01/logs/anthropics__claude-agent-sdk-python__91315e__22fa9f/stdout.log) | [stderr](<experiment_root>/artifacts/swe_ci_runs/nir_pair30_qwen36_triage_max5/assisted/chunk_01/logs/anthropics__claude-agent-sdk-python__91315e__22fa9f/stderr.log) |  |
| argoproj-labs__hera__7ac8d5__3b30d5 | timeout |  |  |  | 7200.188 |  |  | -15 |  | 0 | 0 | [stdout](<experiment_root>/artifacts/swe_ci_runs/nir_pair30_qwen36_triage_max5/assisted/chunk_02/logs/argoproj-labs__hera__7ac8d5__3b30d5/stdout.log) | [stderr](<experiment_root>/artifacts/swe_ci_runs/nir_pair30_qwen36_triage_max5/assisted/chunk_02/logs/argoproj-labs__hera__7ac8d5__3b30d5/stderr.log) | Process timed out after 7200 seconds. |
| cle-b__httpdbg__22e489__af88c4 | success | 5 | -1 | 5 | 2045.547 | 160483 | 42941 | 0 |  | 0 | 5 | [stdout](<experiment_root>/artifacts/swe_ci_runs/nir_pair30_qwen36_triage_max5/assisted/chunk_02/logs/cle-b__httpdbg__22e489__af88c4/stdout.log) | [stderr](<experiment_root>/artifacts/swe_ci_runs/nir_pair30_qwen36_triage_max5/assisted/chunk_02/logs/cle-b__httpdbg__22e489__af88c4/stderr.log) |  |
| cloudtools__troposphere__14a8b3__7ab9bc | success | 5 | -1 | 12 | 1376.371 | 271578 | 48351 | 0 |  | 0 | 5 | [stdout](<experiment_root>/artifacts/swe_ci_runs/nir_pair30_qwen36_triage_max5/assisted/chunk_02/logs/cloudtools__troposphere__14a8b3__7ab9bc/stdout.log) | [stderr](<experiment_root>/artifacts/swe_ci_runs/nir_pair30_qwen36_triage_max5/assisted/chunk_02/logs/cloudtools__troposphere__14a8b3__7ab9bc/stderr.log) |  |
| dbrattli__expression__8a6e63__3981fd | success | 5 | 96 | 43 | 867.362 | 254178 | 31808 | 0 |  | 0 | 5 | [stdout](<experiment_root>/artifacts/swe_ci_runs/nir_pair30_qwen36_triage_max5/assisted/chunk_02/logs/dbrattli__expression__8a6e63__3981fd/stdout.log) | [stderr](<experiment_root>/artifacts/swe_ci_runs/nir_pair30_qwen36_triage_max5/assisted/chunk_02/logs/dbrattli__expression__8a6e63__3981fd/stderr.log) |  |
| desgeeko__pdfsyntax__8fa6b3__a08472 | success | 5 | -1 | 33 | 1407.278 | 226848 | 51265 | 0 |  | 0 | 5 | [stdout](<experiment_root>/artifacts/swe_ci_runs/nir_pair30_qwen36_triage_max5/assisted/chunk_02/logs/desgeeko__pdfsyntax__8fa6b3__a08472/stdout.log) | [stderr](<experiment_root>/artifacts/swe_ci_runs/nir_pair30_qwen36_triage_max5/assisted/chunk_02/logs/desgeeko__pdfsyntax__8fa6b3__a08472/stderr.log) |  |
| ebodshojaei__bake__8de354__a3736b | success | 5 | 54 | 12 | 1134.167 | 274328 | 45347 | 0 |  | 0 | 5 | [stdout](<experiment_root>/artifacts/swe_ci_runs/nir_pair30_qwen36_triage_max5/assisted/chunk_03/logs/ebodshojaei__bake__8de354__a3736b/stdout.log) | [stderr](<experiment_root>/artifacts/swe_ci_runs/nir_pair30_qwen36_triage_max5/assisted/chunk_03/logs/ebodshojaei__bake__8de354__a3736b/stderr.log) |  |

## Errors

- `argoproj-labs__hera__7ac8d5__3b30d5`: Process timed out after 7200 seconds.

## Conclusion

At least one SWE-CI task failed; inspect logs before using this run as a benchmark signal.
