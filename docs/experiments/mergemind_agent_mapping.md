# MergeMind Agent Mapping

This note fixes the current agent map before running SWE-CI and real PR experiments.

## Current Chain

| Component | Current role | Closest proposed tool | Covered well | Weak or missing |
| --- | --- | --- | --- | --- |
| `LLMGenerator` | Generates candidate review comments from MR title, description, repository context, changed files, and unified diff. | `Patch Risk Auditor` plus a weak `Bug Hunter` | Broad recall over correctness, API behavior, edge cases, missing tests, maintainability, and style when useful. | Not SWE-CI-specific; style and maintainability can compete with repair-impact comments. |
| `LLMReranker` | Scores and selects candidate comments by usefulness, groundedness, actionability, and specificity. | `Two-stage Critic` | Strong filtering layer; penalizes generic, speculative, nice-to-have, and saturated local-model scores. | Does not explicitly rank by expected test-gap reduction unless the prompt profile says so. |
| `LLMRewriter` | Rewrites selected comments into concise human-facing code-review feedback. | Weak `Minimal Fix Coach` | Preserves meaning, adds essence/severity/confidence, improves readability. | Does not produce an explicit repair contract for an automated programmer. |
| `LLMJudge` | Optional evaluator for generated comments. | Quality evaluator, not a reviewer tool | Useful for no-gold and gold-comment PR scoring. | Should not feed hints back into SWE-CI or production review generation. |

## Proposed Tool Coverage

- `Patch Risk Auditor`: mostly covered by `LLMGenerator`, strengthened by `qwen35_rewriter_sweci_contract`.
- `Bug Hunter`: partially covered by `LLMGenerator`, strengthened by correctness-only prompts.
- `Two-stage Critic`: covered by `LLMReranker`.
- `Minimal Fix Coach`: weakly covered by `LLMRewriter`; strengthened by contract rewrite prompts.
- `Revision Contract`: newly covered by `SWEContractLLMRewriter`, which emits `Finding`, `Evidence`, `Expected revision`, and `Do not change`.
- `Test Failure Triage Reviewer`: still not fully covered. SWE-CI assisted examples include requirement and programmer diff, but not full non-passed test details as first-class prompt context.
- `No-Target Oracle Guard`: partially covered. SWE-CI assisted artifacts set `target_sha_used_for_review=false`, omit target code, and the contract prompts forbid hidden-solution assumptions.

## Experimental Profile

`qwen35_rewriter_sweci_contract` keeps the same three-agent shape but changes intent:

- Generator: source-risk comments only; no style-only feedback.
- Reranker: score by likely next-revision repair impact and expected failing-gap reduction.
- Rewriter: convert comments into compact agent-facing repair guidance.

Token limits are intentionally modest:

- `max_tokens_contract_generator: 1200`
- `max_tokens_contract_reranker: 900`
- `max_tokens_contract_rewriter: 1200`

These limits are higher than the default rewriter output only where the new contract format needs extra structure. They should be increased only when logs show truncation or parse failures.
