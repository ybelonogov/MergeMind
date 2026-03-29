# Datasets for MergeMind

## Main datasets
### 1. CodeReviewer
Use as the main starting dataset for code review tasks.
Source:
- Zenodo — https://zenodo.org/records/6900648
- comment generation split — `Comment_Generation.zip`
Why it matters:
- directly related to automated code review
- includes review-oriented task formulation
- good baseline for early MVP experiments

### 2. CodeReviewQA
Use for validation of review understanding.
Source:
- Hugging Face — https://huggingface.co/datasets/Tomo-Melb/CodeReviewQA
- note: gated dataset, requires accepting the access conditions
Why it matters:
- checks whether the model understands review reasoning
- better suited for evaluation than for main training

## Auxiliary datasets
### 3. CoDocBench
Use as an auxiliary dataset for change alignment and structural parsing experiments.
Source:
- GitHub — https://github.com/kunpai/codocbench
- dataset files — `dataset/train.jsonl`, `dataset/test.jsonl`, `dataset/codocbench.jsonl`
Why it matters:
- links code changes with related artifacts
- helpful for diff/entity alignment ideas

## Iteration 2 / optional
### 4. SWE-bench
Use only for later experiments that move from comment generation
toward more agentic issue / fix workflows.

## Internal / project-specific data
If available, add a custom dataset built from real MR history:
- MR description
- diff
- review threads
- follow-up commits
- resolved / unresolved outcome

This custom dataset can later become the main source of supervision
for usefulness and actionability.
