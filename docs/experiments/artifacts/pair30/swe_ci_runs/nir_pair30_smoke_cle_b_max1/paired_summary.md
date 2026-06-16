# SWE-CI Baseline vs MergeMind Assisted

- baseline: `<experiment_root>/artifacts/swe_ci_runs/nir_pair30_smoke_cle_b_max1/baseline/all`
- assisted: `<experiment_root>/artifacts/swe_ci_runs/nir_pair30_smoke_cle_b_max1/assisted/all`
- tasks: 1
- compared tasks: 1
- mean iteration delta: 0.000
- mean final gap delta:
- mean same/lower gap iteration delta:
- mean iterations-to-best-gap delta: 0.000
- mean failed-set Jaccard vs baseline: 1.000
- mean official EvoScore delta: 0.000
- mean official solved-rate delta: 0.000
- improved/worse/unchanged/incomplete: 0 / 0 / 1 / 0
- fixed failures: 0
- new failures: 0
- assisted comments: 1
- assisted revisions: 1
- baseline total tokens: 14843.000
- assisted total tokens: 31691.000
- assisted review tokens: 9010.000
- assisted LLM calls: 3.000
- invalid assisted final gaps: 1
- assisted invalid iterations: 1

Negative deltas mean the assisted run used fewer iterations or ended with a smaller valid gap.
Rows with invalid final gaps are excluded from final-gap delta averages.

| task_id | label | baseline | assisted | base_iter | assist_iter | iter_delta | base_gap | assist_gap | gap_delta | base_best_iter | assist_best_iter | best_iter_delta | evoscore_delta | solved_delta | invalid_iters | first_same | first_same/lower | failed_jaccard | fixed | new | same_tests | tokens | review_tokens | comments | revisions |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| cle-b__httpdbg__22e489__af88c4 | unchanged | success | success | 1.000 | 1.000 | 0.000 |  |  |  | 0 | 0 | 0 | 0.000 | 0.000 | 1 |  |  | 1.000 | 0 | 0 | True | 31691.000 | 9010.000 | 1 | 1 |
