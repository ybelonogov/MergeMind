# Supervisor Action Plan

This document tracks the concrete follow-up items from the supervisor feedback.

## Practical Part

### 1. Repositories and Validation State

Current validation branch:

- `codex/mergemind-caveman-sweci-grid`

Important validation documents:

- `docs/experiments/mergemind_agent_mapping.md`
- `docs/experiments/mergemind_qwen35_rewriter_results.md`
- `docs/experiments/mergemind_caveman_sweci_grid_results.md`
- `docs/experiments/mergemind_test_aware_reviewer.md`

Current reproducibility anchors:

- fixed SWE-CI manifest: `configs/swe_ci_caveman_low_gap_tasks.jsonl`
- holdout SWE-CI manifest without already scored tasks: `configs/swe_ci_caveman_holdout_tasks.jsonl`
- comparison script: `scripts/compare_swe_ci_runs.py`
- grid summary script: `scripts/summarize_swe_ci_grid.py`
- official SWE-CI stdout metric parser: `src/validation/swe_ci/result_parser.py`
- requirement/comment injection analyzer: `scripts/analyze_swe_ci_requirements.py`
- Linux worktree used for runs: `/home/pashab/MergeMind-caveman-grid`
- SWE-CI checkout on server: `/home/pashab/SWE-CI`

Next repository-level step:

- create or update a public/clean experiment branch or repository state that contains only the reproducible runner, configs, and documentation needed to validate the experiments;
- keep large run artifacts outside git, but document their server paths and commands.

### 2. More Projects and More Iterations

Current completed SWE-CI evidence is preliminary and task-limited. The next grid should avoid reprocessing the same tasks where possible.

Planned split:

- keep already used tasks as calibration only:
  - `15r10nk__inline-snapshot__3bb05d__e2b9b2`
  - `cle-b__httpdbg__22e489__af88c4`
- use the remaining low-gap manifest tasks for the next scored run:
  - `igrek51__wat__ecddda__8efafa`
  - `kinnay__nintendoclients__e2b673__72846c`
  - `eliben__pycparser__7f6b34__6ba954`
  - `gusye1234__nano-graphrag__bc175d__9b33a7`

Manifest for this split:

- `configs/swe_ci_caveman_holdout_tasks.jsonl`

Suggested next run matrix:

- baseline without MergeMind;
- `qwen35_rewriter_sweci_triage`;
- `qwen35_rewriter_sweci_safe_triage`;
- `qwen35_rewriter_sweci_test_guard`;
- `qwen35_caveman_top1`;
- optional `qwen35_caveman_direct_top1` if token budget allows.

Suggested iteration setting:

- start with `max_iterations=5` for 2 projects;
- expand to all remaining projects only if the local model and server remain stable.
- after the `igrek51/wat` negative holdout, treat `qwen35_caveman_top1` as risky for this task;
- `qwen35_rewriter_sweci_safe_triage` with `max_revision_epochs=1` was also negative on `igrek51/wat` (`5 -> -1 -> -1`) and should not be expanded until the direct-agent guard fix is included;
- after the first guard fix, a rerun stayed valid for one epoch (`5 -> 5`) but did not improve and then hit malformed JSON in the direct programmer agent;
- do not spend GPU time on a wider grid until the direct-agent JSON replacement path is made more reliable.

Dry-run commands before spending GPU time:

```bash
python scripts/run_swe_ci.py \
  --swe-ci-repo-path /home/pashab/SWE-CI \
  --tasks-path configs/swe_ci_caveman_holdout_tasks.jsonl \
  --output-dir artifacts/swe_ci_runs \
  --run-id holdout_baseline_max5_dryrun \
  --limit 2 \
  --max-iterations 5 \
  --mode baseline \
  --splitting lite \
  --agent-name direct_openai \
  --base-url http://127.0.0.1:1234/v1 \
  --model-name qwen3.6-27b@iq2_xxs \
  --api-key lm-studio \
  --source-data-root /home/pashab/SWE-CI/data \
  --docker-network host \
  --dry-run
```

```bash
python scripts/run_swe_ci.py \
  --swe-ci-repo-path /home/pashab/SWE-CI \
  --tasks-path configs/swe_ci_caveman_holdout_tasks.jsonl \
  --output-dir artifacts/swe_ci_runs \
  --run-id holdout_caveman_top1_max5_dryrun \
  --limit 2 \
  --max-iterations 5 \
  --mode mergemind_assisted \
  --splitting lite \
  --agent-name direct_openai \
  --base-url http://127.0.0.1:1234/v1 \
  --model-name qwen3.6-27b@iq2_xxs \
  --api-key lm-studio \
  --source-data-root /home/pashab/SWE-CI/data \
  --docker-network host \
  --mergemind-config configs/base.yaml \
  --mergemind-pipeline qwen35_caveman_top1 \
  --mergemind-llm-provider local_qwen36_27b_iq2 \
  --mergemind-top-n 1 \
  --mergemind-min-score 0.75 \
  --mergemind-max-revision-epochs 2 \
  --dry-run
```

Metrics to report:

- official SWE-CI `EVOSCORE(GAMMA=1)`;
- official SWE-CI `SOLVED_RATE` and `ZERO_REGRESSION`;
- `actual_iterations`;
- `final_gap`;
- `best_gap`;
- failed-test-set overlap;
- new/fixed failures;
- token usage;
- LLM call count;
- latency.

### 3. requirement.xml and Comment Injection Point

Current assisted flow:

1. SWE-CI architect writes `requirement.xml` for the current epoch.
2. SWE-CI programmer makes the initial code patch.
3. MergeMind receives the current epoch requirement plus the before/after programmer diff.
4. MergeMind writes `mergemind_review.md`.
5. The revision container receives:
   - `/app/requirement.xml`;
   - `/app/mergemind_review.md`;
   - `/app/mergemind_allowed_files.txt`;
   - `/app/mergemind_before_files.md` in before-snapshot variants.
6. SWE-CI programmer performs a revision pass.
7. Pytest runs after that revision.

Code pointers:

- injection patch: `src/validation/swe_ci/assisted.py`
- direct agent context reader: `src/validation/swe_ci/direct_openai_agent_template.py`
- review builder: `src/validation/swe_ci/assist_helper.py`
- artifact analyzer: `scripts/analyze_swe_ci_requirements.py`

Reliability fix from 2026-05-27:

- the direct local-model agent now enforces `/app/mergemind_allowed_files.txt` during MergeMind revision passes;
- placeholder path `src/package/module.py` was removed from the JSON prompt example because the local model copied it literally in a revision response;
- outside-patch revision attempts are now non-retryable guard failures instead of 10 repeated attempts.
- MergeMind revision allow-lists are now restricted to Python source files from the programmer patch; non-code files touched by the programmer are recorded as skipped and are not handed to the revision agent.

Test-aware reviewer update from 2026-05-29:

- new pipeline: `qwen35_rewriter_sweci_test_guard`;
- alias: `qwen35_review_test_guard`;
- visible previous pytest failures are always included in this mode;
- reviewed diffs are restricted to Python source files;
- comments must connect a visible failing test to changed Python source and a minimal safe revision.

Reproducible inspection command:

```bash
python scripts/analyze_swe_ci_requirements.py \
  --run-dir artifacts/swe_ci_runs/<run_id> \
  --output artifacts/swe_ci_runs/<run_id>/requirement_injection_analysis.md \
  --json-output artifacts/swe_ci_runs/<run_id>/requirement_injection_analysis.json
```

Open investigation:

- run the analyzer on completed runs and compare `requirement.xml` across epochs;
- check whether requirement drift changes the usefulness of MergeMind comments;
- use requirement hash/text excerpt per iteration when explaining whether the comment was generated against the same or changed requirement.

## NIR and Article Draft

Required immediate items:

- send defense date to supervisor;
- maintain a draft document with headings even if sections are incomplete.

Created draft documents:

- NIR report draft: `docs/nir/main.tex`
- article draft: `docs/nir/article_draft.tex`
- build/readme notes: `docs/nir/README.md`

The article draft already includes:

- literature overview skeleton;
- goals and tasks;
- proposed comment generation/evaluation approach;
- initial experiment table;
- discussion of different comment types and results.

TODO before sending:

- replace title-page placeholders in `docs/nir/main.tex`;
- verify bibliography format against the required ITMO template;
- insert the actual defense date once known.
