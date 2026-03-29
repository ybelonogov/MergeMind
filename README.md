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
  sample_data/
    raw/
  src/
    config.py
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

## Current baseline
- `src/data` normalizes `CodeReviewer`, `CodeReviewQA`, and `CoDocBench`
  into one MR-centric schema.
- `src/context` parses unified diffs, extracts changed identifiers, and uses
  Tree-sitter when available before falling back to lightweight parsing.
- `src/models` implements a local retrieval generator over `CodeReviewer`
  training examples plus a learned logistic reranker with heuristic fallback.
- `src/validation` computes offline deterministic metrics and supports an
  optional OpenAI-backed LLM judge.
- `src/inference` runs the full `context -> generator -> reranker` flow for
  one MR.

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

## Local commands
Install dependencies first:

```bash
python -m pip install -r requirements.txt
```

Download the real datasets configured in [configs/base.yaml](c:\Users\alex\Desktop\ITMO\MergeMind\configs\base.yaml):

```bash
python scripts/download_datasets.py
```

Then run the full pipeline:

```bash
python scripts/prepare_data.py
python scripts/train_baseline.py
python scripts/evaluate.py
python scripts/demo_mr.py
python -m unittest discover -s tests -v
```

Artifacts are written under `artifacts/`:
- `artifacts/data/` - normalized train/validation/test/demo files
- `artifacts/models/` - retrieval generator and reranker artifacts
- `artifacts/evaluation/` - offline metrics and predictions

## Notes
- The baseline is intentionally simple and fully local.
- `CodeReviewer` is the primary training dataset.
- `CodeReviewQA` and `CoDocBench` are validation-side signals in the MVP.
- `CodeReviewQA` is gated on Hugging Face. To download it, accept the dataset
  terms and set `HF_TOKEN` or `HUGGINGFACE_TOKEN` before running
  `python scripts/download_datasets.py`.
- The default config limits prepared data volume so the pipeline stays runnable
  on a local machine. You can raise those limits in `configs/base.yaml`.
- `PyYAML` is not required; config loading uses the local YAML subset parser
  in `src/config.py`.
