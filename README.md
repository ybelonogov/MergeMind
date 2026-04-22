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
  It also includes local OpenAI-compatible LLM generator, reranker, and
  optional final rewriter components for LM Studio / Qwen experiments.
- `src/validation` computes offline deterministic metrics and supports an
  optional OpenAI-backed LLM judge plus a local LM Studio judge.
- `src/inference` runs the full `context -> generator -> reranker` flow for
  one MR, with an optional `rewriter` step for final human-facing wording.

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

## Local Qwen experiments
Start the LM Studio local server, load the configured Qwen model, then check
that MergeMind can see it:

```bash
python scripts/check_llm.py
```

Run the local LLM pipeline on a small profile:

```bash
python scripts/evaluate.py --pipeline qwen35_full --profile smoke
```

Run the same pipeline with the final rewrite/summarization agent:

```bash
python scripts/evaluate.py --pipeline qwen35_rewriter --profile smoke
```

Run the A/B experiment table:

```bash
python scripts/run_experiments.py --profile smoke
```

Inspect a run as a human-readable review report:

```bash
python scripts/inspect_predictions.py --run qwen35_limit1_after_fix --limit 5
```

If a run contains several pipeline modes, pass one explicitly:

```bash
python scripts/inspect_predictions.py --run <run_id> --mode qwen35_full --limit 5
```

Serve the local monitoring dashboard:

```bash
python scripts/dashboard.py
```

Open `http://127.0.0.1:8765` to watch GPU utilization, GPU memory,
LM Studio model status, experiment progress, latency, token usage, token speed,
parse error rate, cache hit rate, and recent run artifacts.

Available experiment modes:
- `baseline_retrieval_logistic`
- `qwen35_generator_logistic_reranker`
- `retrieval_generator_qwen35_reranker`
- `qwen35_generator_qwen35_reranker`
- `qwen35_full_with_rewriter`
- `qwen35_full_with_qwen35_judge`
- `qwen35_full_with_rewriter_and_qwen35_judge`

`qwen35_full` is accepted as a short alias for
`qwen35_generator_qwen35_reranker`.
`qwen35_rewriter` runs the full Qwen pipeline and rewrites the selected
comments into concise final review feedback with `essence`, `severity`, and
`rewrite_confidence` fields.
`qwen35_rewriter_judge` additionally runs the local Qwen judge on the rewritten
comments.

Artifacts are written under `artifacts/`:
- `artifacts/data/` - normalized train/validation/test/demo files
- `artifacts/models/` - retrieval generator and reranker artifacts
- `artifacts/evaluation/` - offline metrics and predictions
- `artifacts/runs/` - per-run LLM/A-B experiment artifacts
- `artifacts/llm_cache.sqlite` - local LLM response cache

## Notes
- The baseline is intentionally simple and fully local.
- The local LLM path uses LM Studio's OpenAI-compatible local endpoint from
  `configs/base.yaml`; no paid API is required.
- `CodeReviewer` is the primary training dataset.
- `CodeReviewQA` and `CoDocBench` are validation-side signals in the MVP.
- `CodeReviewQA` is gated on Hugging Face. To download it, accept the dataset
  terms and set `HF_TOKEN` or `HUGGINGFACE_TOKEN` before running
  `python scripts/download_datasets.py`.
- The default config limits prepared data volume so the pipeline stays runnable
  on a local machine. You can raise those limits in `configs/base.yaml`.
- `PyYAML` is not required; config loading uses the local YAML subset parser
  in `src/config.py`.
