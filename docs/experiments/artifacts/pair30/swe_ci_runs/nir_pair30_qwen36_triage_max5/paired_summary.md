# SWE-CI Baseline vs MergeMind Assisted

- baseline: `<repo_root>\docs\experiments\artifacts\pair30\swe_ci_runs\nir_pair30_qwen36_triage_max5\baseline\all`
- assisted: `<repo_root>\docs\experiments\artifacts\pair30\swe_ci_runs\nir_pair30_qwen36_triage_max5\assisted\all`
- tasks: 15
- compared tasks: 10
- mean iteration delta: 0.000
- mean final gap delta: 12.000
- mean same/lower gap iteration delta: 0.000
- mean iterations-to-best-gap delta: -0.200
- mean failed-set Jaccard vs baseline: 0.522
- mean official EvoScore delta: -0.041
- mean official solved-rate delta: 0.000
- improved/worse/unchanged/incomplete: 1 / 5 / 5 / 4
- fixed failures: 351
- new failures: 215
- assisted comments: 46
- assisted revisions: 46
- baseline total tokens: 2213434.000
- assisted total tokens: 2461631.000
- assisted review tokens: 379598.000
- assisted LLM calls: 139.000
- invalid assisted final gaps: 10
- assisted invalid iterations: 32

Negative deltas mean the assisted run used fewer iterations or ended with a smaller valid gap.
Rows with invalid final gaps are excluded from final-gap delta averages.

| task_id | label | baseline | assisted | base_iter | assist_iter | iter_delta | base_gap | assist_gap | gap_delta | base_best_iter | assist_best_iter | best_iter_delta | evoscore_delta | solved_delta | invalid_iters | first_same | first_same/lower | failed_jaccard | fixed | new | same_tests | tokens | review_tokens | comments | revisions |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 15r10nk__inline-snapshot__3bb05d__e2b9b2 | worse | success | success | 5.000 | 5.000 | 0.000 | 13 | 106 | 93 | 4 | 3 | -1 | -0.216 | 0.000 | 0 | 4 | 3 | 0.735 | 6 | 103 | False | 257268.000 | 25078.000 | 4 | 4 |
| 9001__copyparty__452592__4bdcbc | unchanged | success | success | 5.000 | 5.000 | 0.000 |  | 16 |  | 0 | 0 | 0 | 0.200 | 0.000 | 4 |  |  | 0.000 | 0 | 16 | False | 267706.000 | 43000.000 | 5 | 5 |
| aio-libs__yarl__576b5e__857930 | unchanged | success | success | 5.000 | 5.000 | 0.000 |  |  |  | 0 | 0 | 0 | 0.000 | 0.000 | 5 |  |  | 1.000 | 0 | 0 | True | 251332.000 | 35648.000 | 5 | 5 |
| amaranth-lang__amaranth__46e0a0__a4402b | worse | success | success | 5.000 | 5.000 | 0.000 | 91 |  |  | 0 | 0 | 0 | -0.181 | 0.000 | 3 |  | 0 | 0.000 | 239 | 0 | False | 285181.000 | 45681.000 | 5 | 5 |
| anthropics__claude-agent-sdk-python__91315e__22fa9f | worse | success | success | 5.000 | 5.000 | 0.000 | 40 | 40 | 0 | 1 | 0 | -1 | -0.037 | 0.000 | 0 | 2 | 0 | 1.000 | 0 | 0 | True | 212729.000 | 10479.000 | 2 | 2 |
| argoproj-labs__hera__7ac8d5__3b30d5 | unchanged | timeout | timeout |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  |  |  | False |  |  |  |  |
| cle-b__httpdbg__22e489__af88c4 | unchanged | success | success | 5.000 | 5.000 | 0.000 |  |  |  | 0 | 0 | 0 | 0.000 | 0.000 | 5 |  |  | 1.000 | 0 | 0 | True | 160483.000 | 42941.000 | 5 | 5 |
| cloudtools__troposphere__14a8b3__7ab9bc | worse | success | success | 5.000 | 5.000 | 0.000 |  |  |  | 0 | 0 | 0 | -0.200 | 0.000 | 4 |  |  | 1.000 | 0 | 0 | True | 271578.000 | 48351.000 | 5 | 5 |
| dbrattli__expression__8a6e63__3981fd | unchanged | success | success | 5.000 | 5.000 | 0.000 |  | 96 |  | 0 | 0 | 0 | 0.000 | 0.000 | 4 |  |  | 0.000 | 0 | 96 | False | 254178.000 | 31808.000 | 5 | 5 |
| desgeeko__pdfsyntax__8fa6b3__a08472 | worse | success | success | 5.000 | 5.000 | 0.000 | 49 |  |  | 2 | 2 | 0 | 0.026 | 0.000 | 3 | 1 | 0 | 0.000 | 49 | 0 | False | 226848.000 | 51265.000 | 5 | 5 |
| ebodshojaei__bake__8de354__a3736b | improved | success | success | 5.000 | 5.000 | 0.000 | 111 | 54 | -57 | 0 | 0 | 0 | 0.003 | 0.000 | 4 |  | 0 | 0.486 | 57 | 0 | False | 274328.000 | 45347.000 | 5 | 5 |
| eliben__pycparser__7f6b34__6ba954 | incomplete | success |  | 5.000 |  |  |  |  |  | 0 |  |  |  |  | 0 |  |  |  |  |  | False |  |  |  |  |
| falconry__falcon__a12995__60ecec | incomplete | success |  | 5.000 |  |  |  |  |  | 0 |  |  |  |  | 0 |  |  |  |  |  | False |  |  |  |  |
| grahamdumpleton__wrapt__3125a5__d8803a | incomplete | success |  | 5.000 |  |  |  |  |  | 0 |  |  |  |  | 0 |  |  |  |  |  | False |  |  |  |  |
| gusye1234__nano-graphrag__bc175d__9b33a7 | incomplete | failed |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  |  |  | False |  |  |  |  |
