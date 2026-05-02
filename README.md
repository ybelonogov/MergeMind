# MergeMind

MergeMind — MVP-система для автоматизированного ревью merge request.
Она принимает MR/diff, извлекает контекст изменений, генерирует несколько
кандидатных review-комментариев, ранжирует их и оставляет только наиболее
полезные.

## Цель

Построить локально запускаемый пайплайн, который по контексту MR выдает
1-3 комментария с высокой вероятностью практической пользы для ревью.

Главная идея проекта: не генерировать как можно больше замечаний, а отбирать
небольшое число конкретных, обоснованных и применимых комментариев.

## Структура проекта

```text
MergeMind/
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

## Основной пайплайн

1. **Данные**
   Загрузка и нормализация истории MR, diff, review comments, follow-up commits
   и дополнительных сигналов результата.
2. **Обработка контекста**
   Парсинг diff, выделение измененных файлов, hunks, идентификаторов и
   ближайшего repository context.
3. **Модели**
   Генерация кандидатных комментариев и ранжирование по полезности,
   привязке к diff и конкретности.
4. **Валидация**
   Оценка через deterministic-метрики, LLM judge, latency/runtime-метрики и
   ручной просмотр примеров.

## Что входит в MVP

- подготовка единого формата MR-примеров;
- локальный baseline `generator -> reranker`;
- LLM-пайплайн через LM Studio / Qwen;
- offline evaluation на validation/test;
- demo inference на одном MR;
- dashboard для мониторинга запусков, GPU/LM Studio и артефактов.

## Текущее состояние

- `src/data` нормализует `CodeReviewer`, `CodeReviewQA` и `CoDocBench` в
  единую MR-centric schema.
- `src/context` парсит unified diff, извлекает измененные файлы, hunks,
  добавленные/удаленные строки и changed identifiers. Tree-sitter используется
  как best-effort, при ошибках есть легкий fallback.
- `src/models` содержит retrieval baseline по историческим примерам
  `CodeReviewer`, logistic reranker с heuristic fallback, а также локальные
  LLM-компоненты generator/reranker/rewriter для LM Studio / Qwen.
- `src/validation` считает similarity, hit@k, MRR, runtime-метрики и
  поддерживает LLM judge с оценками `gold_alignment`, `valid_alternative`,
  `groundedness`, `usefulness`.
- `src/inference` собирает полный flow `context -> generator -> reranker`
  и опциональный rewrite step для финального человекочитаемого текста.

## Установка

```bash
python -m pip install -r requirements.txt
```

## Подготовка данных

Скачать реальные датасеты, описанные в [configs/base.yaml](configs/base.yaml):

```bash
python scripts/download_datasets.py
```

Подготовить локальные артефакты:

```bash
python scripts/prepare_data.py
```

После подготовки создаются train/validation/test/demo файлы в `artifacts/data/`.

## Baseline без LLM

```bash
python scripts/train_baseline.py
python scripts/evaluate.py
python scripts/demo_mr.py
```

Этот режим полностью локальный и не требует LM Studio или внешнего API.

## Локальные Qwen-эксперименты

1. Запустить LM Studio Local Server.
2. Загрузить нужную модель, например `qwen/qwen3.5-9b`.
3. Проверить, что проект видит локальный endpoint:

```bash
python scripts/check_llm.py
```

Запуск LLM-пайплайна на маленьком smoke-профиле:

```bash
python scripts/evaluate.py --pipeline qwen35_full --profile smoke
```

Запуск пайплайна с rewrite step:

```bash
python scripts/evaluate.py --pipeline qwen35_rewriter --profile smoke
```

Запуск пайплайна с rewrite step и локальным judge:

```bash
python scripts/evaluate.py --pipeline qwen35_rewriter_judge --profile smoke
```

## Qwen Cloud / DashScope

Ключи хранятся только в `.env`, который не попадает в git:

```bash
DASHSCOPE_API_KEY=...
QWEN_API_KEY=...
```

Проверка облачного провайдера:

```bash
python scripts/check_llm.py --llm-provider qwen_cloud --chat
```

Пример запуска:

```bash
python scripts/evaluate.py --pipeline qwen35_rewriter_judge --llm-provider qwen_cloud --limit 3
```

## Qwen 3.6 27B через LM Studio

Если в LM Studio загружена модель `qwen3.6-27b@iq2_xxs`, можно выбрать
соответствующий provider:

```bash
python scripts/check_llm.py --llm-provider local_qwen36_27b_iq2 --chat
python scripts/run_experiments.py --profile smoke --run-id qwen36_27b_iq2_same_smoke_01 --llm-provider local_qwen36_27b_iq2 --modes baseline_retrieval_logistic qwen35_full qwen35_rewriter qwen35_rewriter_judge
```

Названия режимов `qwen35_*` описывают схему пайплайна, а не жестко зашитую
версию модели. Фактическая модель сохраняется в `run_manifest.json`,
`config_snapshot.json` и `metrics_table.json`.

## GitHub PR demo

MergeMind может использовать реальный GitHub Pull Request как live-вход:
PR загружается read-only, приводится к `MRExample`, затем прогоняется через
обычный inference pipeline.

В `.env` можно задать токен:

```bash
GITHUB_TOKEN=...
```

Для публичных репозиториев токен не обязателен, но с ним выше rate limit.
Для приватных репозиториев токен должен иметь доступ на чтение репозитория.

Dry-run без публикации комментариев в GitHub:

```bash
python scripts/review_github_pr.py --url https://github.com/OWNER/REPO/pull/123 --pipeline qwen35_rewriter --llm-provider local --limit-comments 3 --judge
```

Baseline-вариант без LLM:

```bash
python scripts/review_github_pr.py --url https://github.com/OWNER/REPO/pull/123 --pipeline baseline_retrieval_logistic
```

Скрипт печатает top comments в консоль и сохраняет артефакты:

- `artifacts/github_pr/<owner>_<repo>_pull_<number>/example.json`;
- `artifacts/github_pr/<owner>_<repo>_pull_<number>/predictions.json`.
- `artifacts/github_pr/<owner>_<repo>_pull_<number>/evaluation.json`;
- `artifacts/github_pr/<owner>_<repo>_pull_<number>/report.md`.

Если в PR уже есть human review comments, они используются как временный gold
для `similarity`, `hit@k` и `gold_alignment`. Если comments нет, judge работает
в live no-gold режиме и оценивает practical usefulness, groundedness и
общую практическую ценность по diff/context.

На этом этапе режим безопасный: он ничего не публикует обратно в GitHub.
Следующий возможный шаг — добавить отдельный флаг для draft review после
ручной проверки качества.

## A/B эксперименты

```bash
python scripts/run_experiments.py --profile smoke
```

Основные режимы:

- `baseline_retrieval_logistic`;
- `qwen35_generator_logistic_reranker`;
- `retrieval_generator_qwen35_reranker`;
- `qwen35_generator_qwen35_reranker`;
- `qwen35_full_with_rewriter`;
- `qwen35_full_with_qwen35_judge`;
- `qwen35_full_with_rewriter_and_qwen35_judge`;
- `qwen35_full`;
- `qwen35_rewriter`;
- `qwen35_rewriter_judge`.

`qwen35_rewriter` переписывает выбранные комментарии в короткий review-style
вид с полями `essence`, `severity`, `rewrite_confidence`.

`qwen35_rewriter_judge` дополнительно запускает LLM judge для оценки
переписанных комментариев.

## Просмотр результатов

Посмотреть predictions одного режима:

```bash
python scripts/inspect_predictions.py --run <run_id> --mode qwen35_full --limit 5
```

Сравнить несколько режимов по одним и тем же MR:

```bash
python scripts/compare_run.py --run <run_id> --modes baseline_retrieval_logistic qwen35_full qwen35_rewriter qwen35_rewriter_judge --limit 10
```

Сформировать markdown-отчет для ручного просмотра:

```bash
python scripts/compare_run.py --run <run_id> --limit 5 --diff-lines 18 --output artifacts/runs/<run_id>/report_examples_for_figjam.md
```

## Dashboard

```bash
python scripts/dashboard.py
```

Открыть:

```text
http://127.0.0.1:8765
```

Dashboard показывает:

- статус LM Studio и выбранной модели;
- GPU utilization и GPU memory;
- список последних runs и артефактов;
- progress активного или последнего эксперимента;
- quality metrics: `hit@k`, `best_similarity`, `MRR`, judge scores;
- runtime metrics: inference latency, judge latency, total wall latency;
- токены, uncached tokens/sec, cache hit rate;
- parse error rate и fallback rate.

## Артефакты

Все runtime-артефакты пишутся в `artifacts/` и не должны попадать в git:

- `artifacts/data/` — нормализованные train/validation/test/demo данные;
- `artifacts/models/` — retrieval index и reranker artifacts;
- `artifacts/evaluation/` — offline predictions и metrics;
- `artifacts/runs/` — A/B runs, manifests, summaries, reports;
- `artifacts/llm_cache.sqlite` — SQLite cache LLM-ответов.

## Тесты

```bash
python -m unittest discover -s tests -v
```

## Важные замечания

- `CodeReviewer` — основной тренировочный источник для MVP.
- `CodeReviewQA` и `CoDocBench` используются как validation-side сигналы и
  вспомогательные источники.
- `CodeReviewQA` gated на Hugging Face: перед скачиванием нужно принять условия
  датасета и задать `HF_TOKEN` или `HUGGINGFACE_TOKEN`.
- Базовая оценка работает без платного API.
- LM Studio используется через локальный OpenAI-compatible endpoint из
  `configs/base.yaml`.
- Дефолтные лимиты подготовки данных специально небольшие, чтобы пайплайн
  запускался на локальной машине.
- `PyYAML` не обязателен: конфиг читается встроенным subset parser в
  `src/config.py`.
