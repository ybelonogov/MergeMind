# MergeMind MVP plan

## Goal
Create a system for merge request review that outputs a small number of useful comments
likely to lead to fixes.

## Core idea
The value of the system is not in generating many comments,
but in producing a few comments with high actionability.

## Main blocks
1. **Datasets**
2. **Context processing**
3. **Models**
4. **Validation**

## 1. Datasets
Use real MR history where possible:
- diff
- review comments
- follow-up commits
- thread outcomes

## 2. Context processing
Turn a raw MR into a structured input:
- parse diff
- map changes to code entities
- retrieve related repository context

## 3. Models
Use a simple baseline pipeline:
- generator produces several candidate comments
- reranker selects the most useful ones
- optional rewriter shortens or clarifies long comments

## 4. Validation
Evaluate comments using several signals:
- LLM judge
- tests / CI signals
- benchmark-based evaluation
- manual inspection of examples

## MVP
- use existing datasets
- implement simple preprocessing
- train or configure a baseline generator
- add reranking
- evaluate offline
- run demo on one MR

## Iteration 2
- more agentic workflow
- stronger verification with CI
- possible fix generation
- evaluation on more realistic end-to-end tasks
