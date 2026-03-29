# Literature basis for MergeMind

## Core task and data
- **CodeReviewer** — baseline formulation for automated code review; tasks include quality estimation, comment generation, and code refinement.
- **Desiview / Distilling Desired Comments** — useful for aligning the model on desired review comments instead of all review noise.

## Context processing
- **CoDocBench / Tree-Sitter** — useful for structural parsing of changes and mapping edits to code entities.

## Validation
- **G-Eval** — rubric-based LLM judge for automatic evaluation.
- **CodeReviewQA** — benchmark for understanding review reasoning, not just text generation.
- **METAMON** — verifier-style idea using tests/specification consistency.
- **SWE-CI** — benchmark for more iterative CI-like workflows.
- **Automated Code Review in Practice** — motivates reranking and noise reduction.
- **Towards Practical Defect-Focused Automated Code Review** — close to the target system design for later iterations.

## Reading priority
1. CodeReviewer
2. Desiview
3. CoDocBench / Tree-Sitter
4. G-Eval
5. CodeReviewQA
6. Automated Code Review in Practice
7. METAMON
8. SWE-CI
9. Defect-Focused Automated Code Review
