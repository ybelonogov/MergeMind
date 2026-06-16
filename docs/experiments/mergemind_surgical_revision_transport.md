# MergeMind Surgical Revision Transport

Date: 2026-05-30.

## Goal

The previous SWE-CI runs showed that MergeMind can generate useful review comments, but
the revision pass may apply too broad a file rewrite or spend many retries after a bad
LLM response. This experiment tests safer revision transports for the `direct_openai`
agent.

## Implementation

The direct OpenAI-compatible SWE-CI agent now has two guarded revision transports when
`/app/mergemind_allowed_files.txt` is present:

- `surgical_edits`: asks for exact `old -> new` snippets and applies only one-match
  replacements in allowed source files.
- `bounded_replacement`: asks for full replacement content, but rejects a file if the
  resulting diff exceeds `SWE_CI_DIRECT_REVISION_MAX_CHANGED_LINES`.

Both transports are restricted to the files changed by the programmer patch and still
forbid tests. Revision parse/apply failures are now recorded as `revision_error` with
`changed_files=[]` instead of causing up to 10 repeated revision attempts.

## Validation

Local:

```bash
python -m unittest \
  tests.test_direct_openai_agent_template \
  tests.test_swe_ci_assisted_workdir \
  tests.test_pipeline_modes \
  tests.test_swe_ci_assist_helper
```

Result: `25` tests passed.

Server:

```bash
cd /home/pashab/MergeMind-caveman-grid
.venv/bin/python -m unittest \
  tests.test_direct_openai_agent_template \
  tests.test_swe_ci_assisted_workdir \
  tests.test_pipeline_modes \
  tests.test_swe_ci_assist_helper
```

Result: `25` tests passed.

## Smoke Runs

Task:

- `kinnay__nintendoclients__e2b673__72846c`

Context from previous completed runs:

| run | gap sequence | note |
| --- | --- | --- |
| baseline max3 | `7 -> 27 -> -1 -> -1` | first programmer patch regressed strongly |
| previous test_guard max3 | `7 -> 10 -> -1 -> -1` | only prior positive EvoScore signal on this task |

### Surgical Edits

Run id:

- `surgical_test_guard_kinnay_max3_001`

Command shape:

```bash
python scripts/run_swe_ci.py \
  --tasks-path /tmp/sweci_kinnay_task.jsonl \
  --run-id surgical_test_guard_kinnay_max3_001 \
  --max-iterations 3 \
  --mode mergemind_assisted \
  --agent-name direct_openai \
  --docker-network host \
  --mergemind-pipeline qwen35_rewriter_sweci_test_guard \
  --mergemind-top-n 1 \
  --mergemind-min-score 0.80 \
  --mergemind-max-revision-epochs 1
```

Observed partial result:

| metric | value |
| --- | ---: |
| gap after first epoch | `16` |
| MergeMind comments | `1` |
| MergeMind review tokens | `15902` |
| revision applied | `0` |
| revision error | exact old-text match not found |

The run was stopped after repeated architect failures on epoch 2. This is not a
winning SWE-CI result.

### Bounded Replacement

Run id:

- `bounded_test_guard_kinnay_max1_002`

Command shape:

```bash
SWE_CI_DIRECT_REVISION_TRANSPORT=bounded_replacement \
SWE_CI_DIRECT_REVISION_MAX_CHANGED_LINES=40 \
python scripts/run_swe_ci.py \
  --tasks-path /tmp/sweci_kinnay_task.jsonl \
  --run-id bounded_test_guard_kinnay_max1_002 \
  --max-iterations 1 \
  --mode mergemind_assisted \
  --agent-name direct_openai \
  --docker-network host \
  --mergemind-pipeline qwen35_rewriter_sweci_test_guard \
  --mergemind-top-n 1 \
  --mergemind-min-score 0.80 \
  --mergemind-max-revision-epochs 1
```

Result:

| metric | value |
| --- | ---: |
| gap sequence | `7 -> -1` |
| actual iterations | `1` |
| best valid gap | `7` |
| final gap valid | `false` |
| MergeMind comments | `1` |
| MergeMind revisions | `1` no-op |
| total tokens | `95691` |
| MergeMind review tokens | `18311` |
| MergeMind LLM calls | `3` |

MergeMind comment:

```text
Failing test: tests/switch/test_aauth.py::test_challenge_2000[asyncio]
Finding: The AAuthClient lacks support for system version 2000.
Expected revision: Add version 2000 to the SDK map with "SDK 20.5.4.0" and ensure the endpoint logic returns "/v5/challenge" for this version.
Confidence: 0.9
```

The comment is useful and grounded in visible failure context, but the programmer's
initial patch changed four files and pytest returned `-1`. The bounded revision pass
did not apply a change because the model returned invalid JSON; the new guard correctly
recorded this as `revision_error` and avoided repeated revision retries.

## Conclusion

This approach improves experiment safety, not SWE-CI score:

- It prevents repeated revision attempts after malformed local-model output.
- It records revision transport and revision errors for later analysis.
- It does not improve `kinnay` on the measured smoke: `final_gap=-1`.

The main remaining issue is earlier than MergeMind revision: the programmer's first
patch can already be too broad. The next promising approach is to constrain the first
programmer pass itself with a source-file allowlist and smaller patch contract, rather
than only constraining the MergeMind revision pass.
