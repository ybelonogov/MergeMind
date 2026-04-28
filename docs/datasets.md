# Датасеты для MergeMind

## Основные датасеты

### 1. CodeReviewer

Основной стартовый датасет для задач automated code review.

Источник:

- Zenodo — https://zenodo.org/records/6900648
- split для comment generation — `Comment_Generation.zip`

Почему важен:

- напрямую связан с automated code review;
- содержит постановку задачи генерации review comments;
- подходит для первого retrieval baseline и early MVP экспериментов.

### 2. CodeReviewQA

Используется для проверки понимания review reasoning.

Источник:

- Hugging Face — https://huggingface.co/datasets/Tomo-Melb/CodeReviewQA
- gated dataset, требуется принять условия доступа.

Почему важен:

- проверяет не только генерацию текста, но и понимание причин ревью;
- больше подходит для validation-side проверки, чем для основного обучения
  generator.

## Вспомогательные датасеты

### 3. CoDocBench

Используется как вспомогательный источник для change alignment и структурного
анализа изменений.

Источник:

- GitHub — https://github.com/kunpai/codocbench
- файлы датасета — `dataset/train.jsonl`, `dataset/test.jsonl`,
  `dataset/codocbench.jsonl`

Почему важен:

- связывает изменения в коде с сопутствующими артефактами;
- полезен для идей diff/entity alignment.

## Следующие итерации

### 4. SWE-bench

Рассматривается только для более поздних экспериментов, где проект перейдет от
генерации review comments к issue localization и fix workflows.

## Внутренние данные проекта

Если появится доступ к реальной истории MR, можно собрать project-specific
датасет:

- описание MR;
- diff;
- review threads;
- follow-up commits;
- resolved/unresolved outcome;
- CI/test signals.

Такой датасет может стать основным источником supervision для usefulness,
groundedness и actionability.
