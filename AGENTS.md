# MergeMind

## Project goal
Build an MVP for LLM-based merge request review.
The system should take MR context and produce a small number of comments
with high probability of being useful and leading to a real fix.

## Main pipeline
1. datasets
2. context processing
3. models
4. validation

## Datasets
- CodeReviewer — main code review dataset
- CodeReviewQA — validation of review understanding
- CoDocBench — auxiliary dataset for change alignment
- SWE-bench — optional, only for iteration 2

## Context processing
- diff parsing
- structural code parsing
- repository context retrieval

## Models
- generator: produces several candidate comments
- reranker: ranks comments by usefulness
- optional rewriter: shortens or rephrases long comments

## Validation
- LLM judge
- tests / CI
- benchmark-based evaluation

## Literature basis
- CodeReviewer — task formulation + dataset
- Desiview — alignment on useful review comments
- CoDocBench / Tree-Sitter — structured change parsing
- G-Eval — LLM-as-a-judge
- CodeReviewQA — validation of review reasoning
- METAMON — verification via tests/spec
- SWE-CI — iterative CI scenario
- Automated Code Review in Practice — need for reranking / noise reduction
- Towards Practical Defect-Focused Automated Code Review — target system direction

## Constraints
- Keep the MVP simple
- Prefer clear Python scripts over abstractions
- Do not introduce extra services unless necessary
- All stages must be runnable locally
- Use the text files in docs/ as the source of truth for the project structure

## Read first
- README.md
- docs/plan.md
- docs/literature.md
- docs/datasets.md

## Definition of done
- data preparation works
- baseline model pipeline works
- evaluation script works
- demo inference on one MR works
- commands are documented in README.md
