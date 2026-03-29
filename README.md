# MergeMind

MergeMind is an MVP for automated merge request review.
It is designed as a context-aware pipeline that reads MR changes,
retrieves relevant repository context, generates candidate review comments,
and ranks them to keep only the most useful ones.

## Goal
Build a system that, given MR context, outputs a small number of comments
with a high probability of being useful and leading to a real fix.

## Project structure
```text
MergeMind/
  AGENTS.md
  README.md
  docs/
    plan.md
    literature.md
    datasets.md
  src/
    data/
    context/
    models/
    validation/
    inference/
  scripts/
  configs/
  tests/
```

## Main pipeline
1. **Datasets**  
   Collect and normalize MR history, review comments, follow-up commits, and outcomes.
2. **Context processing**  
   Parse diffs, map changes to code entities, and retrieve repository context.
3. **Models**  
   Generate candidate comments and rerank them by usefulness.
4. **Validation**  
   Evaluate comments with judge-based scoring, tests/CI signals, and benchmarks.

## MVP scope
- use ready datasets
- prepare a unified training / evaluation format
- build a baseline `generator -> reranker` pipeline
- run offline evaluation
- run demo inference on one MR example

## Iteration 2
- more agentic workflow
- CI loop
- fix suggestions
- stronger benchmarks for end-to-end workflow

## Suggested working style with Codex
Use `AGENTS.md` and the files in `docs/` as the source of truth.
Do not invent a different architecture unless explicitly asked.

Recommended task order:
1. create repository skeleton
2. implement data preparation
3. implement context processing
4. implement baseline models
5. implement evaluation
6. implement demo inference

## Initial commands
Add real commands here once the project environment is set up.

```bash
# placeholder examples
python scripts/prepare_data.py
python scripts/train_baseline.py
python scripts/evaluate.py
python scripts/demo_mr.py
```
