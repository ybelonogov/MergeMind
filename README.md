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

## SWE-CI validation setup

SWE-CI слой нужен как downstream benchmark для проверки гипотезы MergeMind:
сокращает ли review-style comment число итераций, время и ошибки coding-agent
в CI-loop. Поддерживаются два режима:

- `baseline` — контрольный запуск официального `swe_ci.evaluate` без
  комментариев MergeMind.
- `mergemind_review_loop` — после запуска SWE-CI ищет patch/diff, который
  сгенерировал coding-agent, строит MRExample без `target_sha`, прогоняет
  MergeMind review pipeline и сохраняет sidecar-комментарии.

Важно: MergeMind не генерирует комментарии из `target_sha`, потому что это
gold fix и утечка ответа. Честный review-loop использует только patch/diff,
полученный после попытки coding-agent.

### Роли подготовки запуска

Роли ниже описывают зоны ответственности при настройке benchmark, а не
отдельные сервисы в коде:

- **Окружение** — Linux/WSL2, Docker, checkout SWE-CI, зависимости.
- **Данные** — SWE-CI dataset и маленький `tasks_smoke.jsonl`.
- **Запуск** — `setup_swe_ci.py`, `run_swe_ci.py --dry-run`, реальные
  команды `swe_ci.evaluate`.
- **Мониторинг** — `events.jsonl`, stdout/stderr, pid, duration,
  CPU/RAM/GPU snapshots.
- **Отчетность** — `summary.md`, `metrics.json`, `task_results.json`.
- **QA** — повторяемость запуска, отсутствие fake success и silent fallback.

SWE-CI запускается из отдельного checkout официального репозитория:

```bash
git clone https://github.com/SKYLENAGE-AI/SWE-CI.git ../SWE-CI
cd ../SWE-CI
python -m pip install -r requirements.txt
```

SWE-CI рассчитан на Linux/Docker окружение и требует заранее подготовленный
датасет. MergeMind не скачивает автоматически полный SWE-CI датасет, потому что
он тяжелый. Датасет нужно подготовить по инструкции SWE-CI отдельно.

Минимальный `tasks.jsonl` для smoke-run содержит реальные SWE-CI task metadata:

```json
{"task_id":"example-task","repo_name":"owner/repo","repo_url":"https://github.com/owner/repo","current_sha":"...","target_sha":"...","image_sha":"...","test_gap":{},"splitting":"default"}
```

Критичные поля обязательны: `task_id`, `repo_name`, `repo_url` или `url`,
`current_sha`, `target_sha`, `image_sha`, `test_gap`. Дополнительные поля
сохраняются в `metadata`; например `splitting`, `api_key`, `base_url`,
`model_name`, `config_file`, `hf_token` используются как CLI-overrides для
реального `python -m swe_ci.evaluate`.

Для `setup_swe_ci.py` и `--dry-run` модель не запускается. Поля `--base-url`,
`--model-name`, `--api-key` нужны реальному SWE-CI coding-agent на этапе
benchmark run. Если SWE-CI/Docker запущен из WSL и LM Studio открыт в Windows,
часто удобнее использовать `http://host.docker.internal:1234/v1`, а не
`http://localhost:1234/v1`.

Рекомендуемая Windows-схема:

- установить WSL2 Ubuntu и включить Docker Desktop WSL integration;
- держать checkout MergeMind и SWE-CI внутри WSL filesystem, например
  `~/MergeMind` и `~/SWE-CI`, а не на `/mnt/c`;
- LM Studio можно оставить в Windows;
- если `host.docker.internal` не резолвится из WSL, использовать IP Windows
  host из `/etc/resolv.conf`.

Проверка окружения:

```bash
python scripts/setup_swe_ci.py \
  --swe-ci-repo-path ../SWE-CI \
  --tasks-path artifacts/swe_ci/tasks_smoke.jsonl \
  --output-dir artifacts/swe_ci_runs
```

Dry-run печатает реальные команды, но не считается benchmark run:

```bash
python scripts/run_swe_ci.py \
  --swe-ci-repo-path ../SWE-CI \
  --tasks-path artifacts/swe_ci/tasks_smoke.jsonl \
  --output-dir artifacts/swe_ci_runs \
  --run-id sweci_smoke_001 \
  --limit 3 \
  --max-iterations 3 \
  --timeout-seconds 7200 \
  --mode baseline \
  --base-url http://host.docker.internal:1234/v1 \
  --model-name qwen3.6-27b@iq2_xxs \
  --api-key lm-studio \
  --dry-run
```

Dry-run для review-loop дополнительно показывает post-step MergeMind review:

```bash
python scripts/run_swe_ci.py \
  --swe-ci-repo-path ../SWE-CI \
  --tasks-path artifacts/swe_ci/tasks_smoke.jsonl \
  --output-dir artifacts/swe_ci_runs \
  --run-id sweci_review_loop_001 \
  --limit 1 \
  --mode mergemind_review_loop \
  --mergemind-pipeline qwen35_rewriter \
  --mergemind-llm-provider local_qwen36_27b_iq2 \
  --dry-run
```

Реальный smoke-run:

```bash
python scripts/run_swe_ci.py \
  --swe-ci-repo-path ../SWE-CI \
  --tasks-path artifacts/swe_ci/tasks_smoke.jsonl \
  --output-dir artifacts/swe_ci_runs \
  --run-id sweci_smoke_001 \
  --limit 1 \
  --max-iterations 3 \
  --timeout-seconds 7200 \
  --mode baseline \
  --base-url http://host.docker.internal:1234/v1 \
  --model-name qwen3.6-27b@iq2_xxs \
  --api-key lm-studio
```

Реальный smoke-run с MergeMind sidecar review:

```bash
python scripts/run_swe_ci.py \
  --swe-ci-repo-path ../SWE-CI \
  --tasks-path artifacts/swe_ci/tasks_smoke.jsonl \
  --output-dir artifacts/swe_ci_runs \
  --run-id sweci_review_loop_001 \
  --limit 1 \
  --max-iterations 3 \
  --timeout-seconds 7200 \
  --mode mergemind_review_loop \
  --base-url http://host.docker.internal:1234/v1 \
  --model-name qwen3.6-27b@iq2_xxs \
  --api-key lm-studio \
  --mergemind-pipeline qwen35_rewriter \
  --mergemind-llm-provider local_qwen36_27b_iq2 \
  --mergemind-top-n 3
```

Артефакты сохраняются в `artifacts/swe_ci_runs/<run_id>/`:

- `run_config.json`, `tasks.json`, `events.jsonl`;
- `task_results.json`, `metrics.json`, `summary.md`;
- `logs/<task_id>/stdout.log`;
- `logs/<task_id>/stderr.log`;
- `logs/<task_id>/events.jsonl`.

В режиме `mergemind_review_loop` для каждой задачи дополнительно сохраняются:

- `swe_ci_outputs/<task_id>/mergemind_example.json`;
- `swe_ci_outputs/<task_id>/mergemind_comments.json`;
- `swe_ci_outputs/<task_id>/mergemind_review.md`.

Если SWE-CI не сохранил patch/diff coding-agent, MergeMind review помечается
как `skipped`; wrapper не пытается строить комментарии из `target_sha`.

Нормальные ошибки на этапе настройки: нет Docker, SWE-CI repo не установлен,
датасет не скачан, task manifest неполный, официальный SWE-CI не создал
результатный файл. В этих случаях wrapper не делает fallback и не выдумывает
успех: задача помечается `failed` или `timeout`, а причина остается в логах и
`summary.md`.

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
