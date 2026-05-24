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
`model_name`, `agent_name`, `config_file`, `hf_token` используются как
CLI-overrides для реального `python -m swe_ci.evaluate`.

Для `setup_swe_ci.py` и `--dry-run` модель не запускается. Поля `--base-url`,
`--model-name`, `--api-key` нужны реальному SWE-CI coding-agent на этапе
benchmark run. Поддерживаемые backend'ы SWE-CI задаются через `--agent-name`:
`iflow` или `opencode`. Если SWE-CI/Docker запущен из WSL и LM Studio открыт в
Windows, часто удобнее использовать `http://host.docker.internal:1234/v1`, а не
`http://localhost:1234/v1`.

Первый запуск SWE-CI может долго собирать Docker image для agent backend'а
(`Dockerfile.iflow` или `Dockerfile.opencode`). Чтобы не смешивать долгий build
с benchmark run, сначала можно прогреть Docker layer cache отдельной командой:

```bash
python scripts/prebuild_swe_ci_agent.py \
  --swe-ci-repo-path ../SWE-CI \
  --base-image image_pypa__build__ffe5ee__010b6c:latest \
  --agent-name iflow \
  --builder legacy \
  --timeout-seconds 3600
```

Логи prebuild сохраняются в `artifacts/swe_ci_agent_builds/`. Для `iflow`
узкое место обычно `npm install -g @iflow-ai/iflow-cli`; это инфраструктурная
стадия, а не вызов локальной LLM.

Рекомендуемая Windows-схема:

- установить WSL2 Ubuntu и включить Docker Desktop WSL integration;
- держать checkout MergeMind и SWE-CI внутри WSL filesystem, например
  `~/MergeMind` и `~/SWE-CI`, а не на `/mnt/c`;
- LM Studio можно оставить в Windows;
- если `host.docker.internal` не резолвится из WSL, использовать IP Windows
  host из `/etc/resolv.conf`.

Схема с отдельным Linux-компьютером:

- MergeMind, SWE-CI и Docker запускаются на Linux-машине;
- LM Studio остается на Windows-машине с загруженной локальной моделью;
- в LM Studio нужно разрешить входящие подключения по сети, не только
  `localhost`;
- в Windows Firewall нужно открыть входящий TCP-порт `1234`;
- в командах SWE-CI использовать LAN endpoint Windows-машины, например
  `http://192.168.1.50:1234/v1`;
- внутри Docker-контейнеров на Linux тоже используется этот LAN endpoint, а не
  `host.docker.internal`.

Быстрая генерация команд для Linux/LAN-сценария:

```bash
python scripts/prepare_linux_lan_swe_ci.py \
  --windows-host 192.168.1.50 \
  --model-name qwen3.6-27b@iq2_xxs \
  --agent-name iflow
```

Проверка доступности LM Studio с Linux-компа:

```bash
curl http://192.168.1.50:1234/v1/models
```

Если этот `curl` не возвращает список моделей, SWE-CI тоже не сможет вызвать
локальную LLM. В таком случае проверяются LM Studio network binding, Windows
Firewall и доступность Windows-машины по LAN.

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
  --agent-name iflow \
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
  --agent-name iflow \
  --base-url http://host.docker.internal:1234/v1 \
  --model-name qwen3.6-27b@iq2_xxs \
  --api-key lm-studio \
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
  --agent-name iflow \
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
  --agent-name iflow \
  --base-url http://host.docker.internal:1234/v1 \
  --model-name qwen3.6-27b@iq2_xxs \
  --api-key lm-studio \
  --mergemind-pipeline qwen35_rewriter \
  --mergemind-llm-provider local_qwen36_27b_iq2 \
  --mergemind-top-n 3
```

Режим `mergemind_review_loop` является post-hoc проверкой: он сохраняет
комментарии MergeMind после завершения SWE-CI задачи и не может уменьшить
число итераций. Для A/B проверки влияния review comments на loop используется
`mergemind_assisted`.

В `mergemind_assisted` wrapper готовит отдельную instrumented-копию SWE-CI
checkout с тем же исходным кодом benchmark. Baseline может идти на чистом
checkout, а assisted-копия добавляет минимальную вставку в epoch:

```text
architect -> requirement.xml -> programmer -> MergeMind review -> programmer revision -> pytest
```

MergeMind получает только diff между кодом до epoch и patch'ем programmer,
`requirement.xml`, `repo_url` и `current_sha`. `target_sha` не передается в
prompt, example или review artifacts.

Если LM Studio поднят на Windows, а SWE-CI запускается на удаленном Linux,
безопасный вариант — SSH reverse tunnel:

```powershell
ssh -N -R 1234:127.0.0.1:1234 -p 7090 pashab@46.146.231.152
```

На Linux-сервере проверить доступ с host и из Docker:

```bash
curl http://127.0.0.1:1234/v1/models
docker run --rm --network host curlimages/curl http://127.0.0.1:1234/v1/models
```

Smoke A/B на одной и той же задаче:

```bash
python scripts/run_swe_ci.py \
  --swe-ci-repo-path ../SWE-CI \
  --tasks-path artifacts/swe_ci/tasks_smoke.jsonl \
  --output-dir artifacts/swe_ci_runs \
  --run-id sweci_baseline_smoke_001 \
  --limit 1 \
  --max-iterations 3 \
  --timeout-seconds 7200 \
  --mode baseline \
  --agent-name opencode \
  --base-url http://127.0.0.1:1234/v1 \
  --model-name qwen3.6-27b@iq2_xxs \
  --api-key lm-studio \
  --docker-network host

python scripts/run_swe_ci.py \
  --swe-ci-repo-path ../SWE-CI \
  --tasks-path artifacts/swe_ci/tasks_smoke.jsonl \
  --output-dir artifacts/swe_ci_runs \
  --run-id sweci_assisted_smoke_001 \
  --limit 1 \
  --max-iterations 3 \
  --timeout-seconds 7200 \
  --mode mergemind_assisted \
  --agent-name opencode \
  --base-url http://127.0.0.1:1234/v1 \
  --model-name qwen3.6-27b@iq2_xxs \
  --api-key lm-studio \
  --docker-network host \
  --mergemind-pipeline qwen35_rewriter \
  --mergemind-llm-provider local_qwen36_27b_iq2 \
  --mergemind-top-n 3
```

Сравнить результаты:

```bash
python scripts/compare_swe_ci_runs.py \
  --baseline-run-dir artifacts/swe_ci_runs/sweci_baseline_smoke_001 \
  --assisted-run-dir artifacts/swe_ci_runs/sweci_assisted_smoke_001 \
  --output artifacts/swe_ci_runs/sweci_ab_smoke_001.md
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

В режиме `mergemind_assisted` дополнительно сохраняются:

- `workdirs/assisted_swe_ci/` — instrumented copy SWE-CI;
- `mergemind_assist/<task_id>/epoch_*/mergemind_example.json`;
- `mergemind_assist/<task_id>/epoch_*/mergemind_comments.json`;
- `mergemind_assist/<task_id>/epoch_*/mergemind_review.md`;
- `mergemind_assist/<task_id>/epoch_*/helper_stdout.log`;
- `mergemind_assist/<task_id>/epoch_*/helper_stderr.log`.

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

## Monitoring Agent

Для фиксации хода работы и подготовки материалов к отчету есть отдельный
monitoring agent. Он не запускает эксперименты сам, а собирает текущее
состояние проекта, git, LM Studio/GPU, A/B runs и SWE-CI artifacts.

Разовый snapshot:

```bash
python scripts/monitoring_agent.py
```

Snapshot с прогоном тестов:

```bash
python scripts/monitoring_agent.py --run-tests
```

Непрерывная летопись каждые 5 минут:

```bash
python scripts/monitoring_agent.py --watch --interval-seconds 300
```

Артефакты пишутся в `artifacts/monitoring/`:

- `chronicle.md` — накопительная MD-летопись;
- `<snapshot_id>/chronicle.md` — отчет конкретного snapshot;
- `<snapshot_id>/dashboard.html` и `latest_dashboard.html` — статический
  dashboard для открытия в браузере;
- `<snapshot_id>/presentation.md` и `latest_presentation.md` — структура
  короткой презентации по текущему состоянию;
- `snapshot.json`, `latest_snapshot.json`, `heartbeat.json` — machine-readable
  состояние для внешних dashboard/tools.

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
