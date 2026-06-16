# MergeMind

MergeMind — прототип автоматического ревью, который проверяется не только по
тексту комментария, а по тому, как этот комментарий влияет на дальнейшее
исправление кода агентом.

Основной сценарий этой ветки: встроить один выбранный комментарий MergeMind в
цикл SWE-CI и сравнить запуск агента без MergeMind с запуском, где после первой
правки агент получает краткую подсказку ревью.

## Что проверяется

Гипотеза проекта: один конкретный и обоснованный комментарий ревью может помочь
агенту быстрее прийти к рабочему исправлению, уменьшить число падающих тестов
или хотя бы не допустить вредной дополнительной правки.

SWE-CI используется как экспериментальный стенд: он запускает агента на задачах
из реальных репозиториев и дает измеримые результаты по тестам. В этой ветке
SWE-CI важен не как отдельная тема, а как способ проверить MergeMind в
итеративном цикле исправления.

## Как работает MergeMind в SWE-CI

В обычном запуске SWE-CI агент получает требование, меняет код и запускает
тесты. В запуске с MergeMind между первой правкой и тестами добавляется шаг
ревью:

```text
требование SWE-CI
  -> агент делает первую правку
  -> MergeMind смотрит требование, видимый diff и доступный контекст
  -> генератор предлагает комментарий
  -> оценщик принимает или отклоняет комментарий
  -> копирайтер переводит принятый комментарий в короткую инструкцию
  -> агент делает дополнительную правку
  -> запускаются тесты SWE-CI
```

MergeMind не получает `target_sha`, скрытое правильное исправление или
эталонную разницу с ним. Это принципиально: комментарий должен строиться только
на видимой информации, которая была бы доступна в реальном цикле исправления.

## Где смотреть главное

Опираться нужно на эту цепочку файлов:

- список задач: `configs/swe_ci_nir_pair30_tasks.jsonl`;
- код запуска: `scripts/run_swe_ci_pair_chunks.py` и `scripts/run_swe_ci.py`;
- компактная сводка текущего pair30-прогона:
  `docs/experiments/artifacts/pair30/swe_ci_runs/nir_pair30_qwen36_triage_max5/paired_summary.json`;
- человекочитаемая сводка:
  `docs/experiments/artifacts/pair30/swe_ci_runs/nir_pair30_qwen36_triage_max5/paired_summary.md`.

Статья:

- `docs/nir/article_draft_v2.tex`

Код интеграции с SWE-CI:

- `scripts/run_swe_ci.py` — одиночный запуск SWE-CI в режимах `baseline`,
  `mergemind_review_loop` и `mergemind_assisted`;
- `src/validation/swe_ci/assisted.py` — подготовка instrumented-копии SWE-CI;
- `src/validation/swe_ci/assist_helper.py` — генерация комментария MergeMind
  внутри итерации;
- `src/validation/swe_ci/direct_openai_agent_template.py` — передача
  комментария агенту и ограниченная дополнительная правка;
- `scripts/compare_swe_ci_runs.py` — сравнение baseline и MergeMind-assisted;
- `scripts/run_swe_ci_pair_chunks.py` — парный запуск baseline/assisted по
  chunk-файлам;
- `scripts/prepare_swe_ci_pair_manifest.py` — подготовка фиксированного списка
  задач;
- `scripts/combine_swe_ci_run_chunks.py` — объединение chunk-результатов.

Документация по экспериментам:

- `docs/experiments/swe_ci_pair30_run.md` — как запускался pair30-прогон и что
  именно сохранено;
- `docs/experiments/mergemind_nir_sweci_batch_2026_05_30.md` — четырехзадачный
  проверочный запуск;
- `docs/experiments/mergemind_surgical_revision_transport.md` — безопасная
  передача дополнительных правок агенту;
- `docs/experiments/mergemind_test_aware_reviewer.md` — более строгий оценщик
  комментариев;
- `docs/experiments/mergemind_caveman_sweci_grid_results.md` — ранние пробные
  SWE-CI запуски.

Компактные артефакты pair30:

- `docs/experiments/artifacts/pair30/swe_ci_runs/nir_pair30_smoke_cle_b_max1/`
- `docs/experiments/artifacts/pair30/swe_ci_runs/nir_pair30_qwen36_triage_max5/`

Сырые runtime-логи, prompt logs, workdirs и Docker/SWE-CI outputs не хранятся в
git. В репозитории лежат компактные сводки, JSON-результаты и команды, которые
нужны для проверки чисел. У большого pair30-прогона файл `commands.jsonl`
неполный, поэтому для воспроизведения нужно использовать команды из этого
README и `docs/experiments/swe_ci_pair30_run.md`, а не этот файл как
единственный журнал запуска.

## Текущие результаты

Положительный пробный сигнал:

- на задаче `cle-b/httpdbg` MergeMind в одном из smoke-сравнений удержал
  финальный разрыв по тестам на уровне `5`, тогда как baseline завершился с
  разрывом `11`;
- новых финальных падений в этом сравнении не появилось.

Ограничения:

- на `inline-snapshot` MergeMind не уменьшил финальный разрыв и не сократил
  число итераций;
- четырехзадачный проверочный запуск не улучшил средний лучший разрыв;
- частичный pair30-прогон пока не является полным 30-task результатом.

Текущий вывод: MergeMind можно встроить в SWE-CI и на отдельных задачах он
может улучшать ход исправления, но устойчивое сокращение числа итераций пока
не доказано.

## Pair30: что сохранено

Фиксированный список задач:

```text
configs/swe_ci_nir_pair30_tasks.jsonl
```

Текущий сохраненный pair30-набор является частичным:

- baseline: 15 task rows;
- assisted: 11 task rows;
- сравнимые успешные пары: 10;
- среднее изменение числа итераций: `0.000`;
- среднее изменение итерации лучшего результата: `-0.200`;
- среднее изменение official EvoScore: `-0.041`;
- improved / worse / unchanged / incomplete: `1 / 5 / 5 / 4`;
- MergeMind comments: `46`;
- MergeMind revisions: `46`;
- MergeMind review tokens: `379598`.

Главная сводка:

```text
docs/experiments/artifacts/pair30/swe_ci_runs/nir_pair30_qwen36_triage_max5/paired_summary.md
```

Эти числа нельзя описывать как результат полного запуска на 30 задачах.

Артефакты pair30 устроены так:

```text
<run-id>/
  baseline/chunk_XX/
  assisted/chunk_XX/
  baseline/all/
  assisted/all/
  paired_summary.md
  paired_summary.json
```

`baseline` — обычный запуск SWE-CI без MergeMind. `assisted` — запуск, где
MergeMind вставлен между первой правкой агента и запуском тестов.

## Воспроизведение pair30

Подготовить chunk-файлы из фиксированного списка задач:

```bash
python scripts/prepare_swe_ci_pair_manifest.py \
  --source-tasks-path configs/swe_ci_nir_pair30_tasks.jsonl \
  --output-dir configs \
  --output-stem swe_ci_nir_pair30_tasks \
  --limit 30 \
  --chunk-size 5
```

Запустить baseline и MergeMind-assisted по chunk-файлам:

```bash
python scripts/run_swe_ci_pair_chunks.py \
  --swe-ci-repo-path <swe_ci_root> \
  --chunks-dir configs \
  --chunk-glob 'swe_ci_nir_pair30_tasks_chunk_*.jsonl' \
  --output-root artifacts/swe_ci_runs \
  --run-id nir_pair30_qwen36_triage_max5 \
  --source-data-root <swe_ci_data_root> \
  --base-url http://127.0.0.1:1234/v1 \
  --model-name 'qwen3.6-27b@iq2_xxs' \
  --api-key lm-studio \
  --docker-network host \
  --max-iterations 5 \
  --timeout-seconds 7200 \
  --mergemind-pipeline qwen35_rewriter_sweci_triage \
  --mergemind-llm-provider local_qwen36_27b_iq2 \
  --mergemind-top-n 1 \
  --mergemind-min-score 0.75 \
  --mergemind-max-revision-epochs 5
```

Объединить chunk-результаты и построить парное сравнение:

```bash
python scripts/combine_swe_ci_run_chunks.py \
  --chunks-parent artifacts/swe_ci_runs/nir_pair30_qwen36_triage_max5/baseline \
  --output-dir artifacts/swe_ci_runs/nir_pair30_qwen36_triage_max5/baseline/all \
  --run-id baseline_all

python scripts/combine_swe_ci_run_chunks.py \
  --chunks-parent artifacts/swe_ci_runs/nir_pair30_qwen36_triage_max5/assisted \
  --output-dir artifacts/swe_ci_runs/nir_pair30_qwen36_triage_max5/assisted/all \
  --run-id assisted_all

python scripts/compare_swe_ci_runs.py \
  --baseline-run-dir artifacts/swe_ci_runs/nir_pair30_qwen36_triage_max5/baseline/all \
  --assisted-run-dir artifacts/swe_ci_runs/nir_pair30_qwen36_triage_max5/assisted/all \
  --output artifacts/swe_ci_runs/nir_pair30_qwen36_triage_max5/paired_summary.md \
  --json-output artifacts/swe_ci_runs/nir_pair30_qwen36_triage_max5/paired_summary.json
```

## Основные метрики

- `status` и `pass_rate` — успешность wrapper/SWE-CI процесса, а не признак,
  что задача решена;
- `actual_iterations` — сколько итераций выполнил агент;
- `best_gap` — минимальное число падающих тестов за запуск;
- `final_gap` — число падающих тестов в последней итерации;
- `final_gap=-1` — невалидный финальный запуск, это не считается улучшением;
- `official_evoscore` — официальный показатель SWE-CI;
- `fixed_failure_count` и `new_failure_count` — какие финальные падения
  исчезли или появились относительно baseline;
- `mergemind_assist_comment_count` — сколько комментариев MergeMind было
  передано агенту;
- `mergemind_review_tokens` — стоимость работы MergeMind.

Качество решения нужно смотреть по `final_gap`, `best_gap`,
`official_evoscore`, `fixed_failure_count` и `new_failure_count`. Технический
статус запуска сам по себе не доказывает, что агент решил задачу.

## Установка и проверки

Установить зависимости:

```bash
python -m pip install -r requirements.txt
```

Запустить тесты, которые покрывают текущий SWE-CI/pair30 слой:

```bash
python -m unittest \
  tests.test_compare_swe_ci_runs \
  tests.test_direct_openai_agent_template \
  tests.test_llm \
  tests.test_swe_ci_assisted_workdir \
  tests.test_swe_ci_command_config \
  tests.test_combine_swe_ci_run_chunks \
  tests.test_prepare_swe_ci_pair_manifest
```

Последняя локальная проверка этой ветки: `49` тестов, `OK`.

## Что не является главным в этой ветке

В репозитории еще остается старый offline/PR-review код: подготовка MR-примеров,
baseline retrieval/reranker, GitHub PR demo, dashboard и LLM judge. Это
предыдущий слой проекта. Для НИР и текущей проверки главный путь такой:

```text
MergeMind -> комментарий ревью -> SWE-CI assisted run -> сравнение с baseline
```

Если старый код или старые документы упоминают конкретные модели и provider'ы,
это нужно читать как технический способ запустить LLM, а не как цель работы.
