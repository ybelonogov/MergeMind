# Литературная база MergeMind

## Основная задача и данные

- **CodeReviewer** — базовая постановка automated code review. Включает quality
  estimation, comment generation и code refinement.
- **Desiview / Distilling Desired Comments** — полезно для идеи отбора
  действительно желательных review comments, а не всего шума из ревью.

## Обработка контекста

- **CoDocBench / Tree-Sitter** — полезны для структурного парсинга изменений,
  выделения сущностей кода и привязки diff к context retrieval.

## Валидация

- **G-Eval** — пример rubric-based LLM judge для автоматической оценки.
- **CodeReviewQA** — benchmark для понимания review reasoning, а не только
  генерации похожего текста.
- **METAMON** — идея verifier-style проверки через tests/specification
  consistency.
- **SWE-CI** — benchmark для более итеративных CI-like workflows.
- **Automated Code Review in Practice** — мотивация reranking и борьбы с noise.
- **Towards Practical Defect-Focused Automated Code Review** — близко к целевой
  системе для следующих итераций.

## Приоритет чтения

1. CodeReviewer
2. Desiview
3. CoDocBench / Tree-Sitter
4. G-Eval
5. CodeReviewQA
6. Automated Code Review in Practice
7. METAMON
8. SWE-CI
9. Defect-Focused Automated Code Review
