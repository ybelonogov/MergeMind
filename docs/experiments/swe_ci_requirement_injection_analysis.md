# SWE-CI Requirement And MergeMind Injection Analysis

This note documents the concrete evidence for the supervisor question:

- how `requirement.xml` changes between SWE-CI iterations;
- when the MergeMind comment is passed to the SWE-CI programmer agent.

## Tool

Use:

```bash
python scripts/analyze_swe_ci_requirements.py \
  --run-dir artifacts/swe_ci_runs/<run_id> \
  --output artifacts/swe_ci_runs/<run_id>/requirement_injection_analysis.md \
  --json-output artifacts/swe_ci_runs/<run_id>/requirement_injection_analysis.json
```

The script reads `task_results.json`, locates the task `iteration.jsonl`, finds
per-epoch `requirement.xml` files, hashes them, and aligns them with
`mergemind_review` / `programmer_revision` entries.

## Injection Point

In `mergemind_assisted` mode the comment is injected inside the SWE-CI epoch:

1. SWE-CI architect writes `requirement.xml`.
2. SWE-CI programmer makes the initial patch.
3. MergeMind reviews the before/after programmer diff using the current
   requirement and visible context.
4. MergeMind writes `mergemind_review.md`.
5. The revision container receives:
   - `/app/requirement.xml`;
   - `/app/mergemind_review.md`;
   - `/app/mergemind_allowed_files.txt`;
   - `/app/mergemind_before_files.md` for before-snapshot variants.
6. SWE-CI programmer performs a revision pass.
7. Pytest runs after the revision pass.

Code anchors:

- `src/validation/swe_ci/assisted.py`: patches SWE-CI `run.py`, calls
  `run_mergemind_assist(...)`, copies review artifacts into the revision
  container, and records `mergemind_review` / `programmer_revision` in
  `iteration.jsonl`.
- `src/validation/swe_ci/direct_openai_agent_template.py`: reads
  `/app/requirement.xml` and `/app/mergemind_review.md` into the direct agent
  prompt context.
- `src/validation/swe_ci/assist_helper.py`: builds the MergeMind example from
  the requirement and programmer diff.

## Example: cle-b/httpdbg control triage run

Validated against server artifact:

- run: `/home/pashab/MergeMind-caveman-grid/artifacts/swe_ci_runs/caveman_grid_control_triage_cle_b_001`
- task: `cle-b__httpdbg__22e489__af88c4`
- requirement files found: `3`

Summary from the analyzer:

| iter | epoch | gap | passed | requirement changed | review status | comments | apply revision | revision present | target sha used |
| ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- | --- |
| 0 | 0 | 5 | 19 | n/a | n/a | 0 | n/a | false | n/a |
| 1 | 1 | 5 | 19 | n/a | skipped | 0 | false | false | false |
| 2 | 2 | 5 | 19 | true | success | 1 | true | true | false |
| 3 | 3 | 5 | 19 | true | skipped | 0 | false | false | false |

Interpretation:

- `requirement.xml` changed between epochs in this run.
- The successful MergeMind comment happened at epoch 2.
- The comment was applied before pytest because `programmer_revision_present`
  is true in the same iteration row.
- `target_sha_used_for_review=false`, so the review did not use hidden target
  commit information.

This gives a concrete answer to the current integration question: MergeMind's
comment is passed to the SWE-CI programmer as revision context after the
programmer's first patch and before test execution.
