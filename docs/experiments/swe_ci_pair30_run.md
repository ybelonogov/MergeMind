# SWE-CI pair30: baseline против MergeMind

Этот документ описывает проверку MergeMind на фиксированном списке из 30 задач
SWE-CI. Цель запуска — сравнить обычный режим SWE-CI с режимом, где после
первой правки агент получает один комментарий MergeMind и делает дополнительную
правку перед запуском тестов.

## Режимы сравнения

- `baseline` — обычный SWE-CI запуск без MergeMind.
- `assisted` — SWE-CI запуск с MergeMind между первой правкой агента и запуском
  тестов.

Оба режима используют один и тот же список задач и одинаковый лимит итераций.
В текущем запуске лимит равен `5`.

`direct_openai` — это backend агента для OpenAI-compatible endpoint. В текущих
прогонах он использовался для локального/удаленного доступа к модели. Это
инфраструктурная деталь, а не отдельная цель эксперимента.

## Что хранится в репозитории

Фиксированный список задач:

```text
configs/swe_ci_nir_pair30_tasks.jsonl
```

Компактные артефакты:

```text
docs/experiments/artifacts/pair30/swe_ci_runs/nir_pair30_smoke_cle_b_max1/
docs/experiments/artifacts/pair30/swe_ci_runs/nir_pair30_qwen36_triage_max5/
docs/experiments/artifacts/pair30/swe_ci_runs/nir_pair30_qwen36_triage_max5/paired_summary.md
docs/experiments/artifacts/pair30/swe_ci_runs/nir_pair30_qwen36_triage_max5/paired_summary.json
```

Сырые workdirs, Docker outputs, stdout/stderr и prompt logs не коммитятся. В
git лежат только компактные сводки и JSON-результаты, которые нужны для
проверки чисел в статье.

Важно: у большого прогона `commands.jsonl` неполный. Воспроизводить запуск надо
по командам ниже, а не по этому файлу как единственному журналу.

## Подготовка задач

Перед запуском chunk-файлы можно пересоздать из общего списка:

```bash
python scripts/prepare_swe_ci_pair_manifest.py \
  --source-tasks-path configs/swe_ci_nir_pair30_tasks.jsonl \
  --output-dir configs \
  --output-stem swe_ci_nir_pair30_tasks \
  --limit 30 \
  --chunk-size 5
```

Скрипт создает:

```text
configs/swe_ci_nir_pair30_tasks_chunk_01.jsonl
...
configs/swe_ci_nir_pair30_tasks_chunk_06.jsonl
configs/swe_ci_nir_pair30_tasks_manifest_info.json
```

Chunk-файлы являются производными от
`configs/swe_ci_nir_pair30_tasks.jsonl`; их можно регенерировать.

## Запуск pair30

Если модель доступна через LM Studio на Windows, а SWE-CI запускается на Linux
сервере, нужен reverse tunnel:

```powershell
ssh -N -R 1234:127.0.0.1:1234 -p 7090 user@server
```

Основной запуск:

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

`scripts/run_swe_ci_pair_chunks.py` запускает каждый chunk дважды:
`baseline` и `assisted`. После этого результаты объединяются и сравниваются.

Итоговая структура runtime-артефактов:

```text
artifacts/swe_ci_runs/<run-id>/
  baseline/chunk_XX/
  assisted/chunk_XX/
  baseline/all/
  assisted/all/
  paired_summary.md
  paired_summary.json
```

## Текущий сохраненный результат

Сейчас в репозитории сохранен частичный pair30-прогон по первым chunk'ам:

- baseline combined tasks: `15`;
- assisted combined tasks: `11`;
- compared successful task pairs: `10`;
- mean iteration delta: `0.000`;
- mean iterations-to-best-gap delta: `-0.200`;
- mean official EvoScore delta: `-0.041`;
- improved / worse / unchanged / incomplete: `1 / 5 / 5 / 4`;
- assisted comments: `46`;
- assisted revisions: `46`;
- assisted review tokens: `379598`.

Эти числа нельзя описывать как результат полного запуска на 30 задачах.

## Метрики

- `actual_iterations` — сколько итераций выполнил агент.
- `best_gap` — минимальное число падающих тестов за запуск.
- `final_gap` — число падающих тестов в последней итерации.
- `final_gap=-1` — невалидный финальный запуск; это не улучшение.
- `official_evoscore` — официальный показатель SWE-CI.
- `fixed_failure_count` — сколько финальных падений baseline исчезло.
- `new_failure_count` — сколько новых финальных падений появилось.
- `mergemind_assist_comment_count` — сколько комментариев MergeMind передано
  агенту.
- `mergemind_review_tokens` — стоимость работы MergeMind.

`status` и `pass_rate` показывают, что wrapper/SWE-CI процесс отработал и был
распарсен. Они не означают, что задача решена.

## Что нельзя смешивать

Положительный smoke-сигнал `cle-b/httpdbg` с финальным разрывом `11 -> 5`
относится к ранней серии пробных SWE-CI запусков, а не к текущему частичному
pair30-артефакту. В pair30 этот результат не надо выдавать за строку из
`paired_summary.json`.

Текущий pair30 нужен для более честной проверки: он показывает, что один
положительный пример не доказывает устойчивое снижение числа итераций или
улучшение качества на наборе задач.
