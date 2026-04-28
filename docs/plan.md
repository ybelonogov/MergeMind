# План MVP MergeMind

## Цель

Создать систему для ревью merge request, которая выдает небольшое число
полезных комментариев, способных привести к реальному исправлению.

## Основная идея

Ценность системы не в количестве сгенерированных замечаний, а в том, чтобы
найти несколько конкретных, обоснованных и применимых комментариев.

## Основные блоки

1. **Данные**
2. **Обработка контекста**
3. **Модели**
4. **Валидация**

## 1. Данные

По возможности используются реальные данные code review:

- diff;
- review comments;
- follow-up commits;
- outcome thread или MR;
- дополнительные метаданные.

## 2. Обработка контекста

Raw MR переводится в структурированный вход:

- парсинг diff;
- выделение измененных файлов и hunks;
- извлечение добавленных/удаленных строк;
- поиск changed identifiers;
- retrieval ближайшего repository context.

Tree-sitter используется как best-effort-обогащение. Если структурный парсинг
не сработал, пайплайн продолжает работу на уровне файла и hunk.

## 3. Модели

Минимальный baseline:

- generator предлагает несколько candidate review comments;
- reranker выбирает наиболее полезные;
- optional rewrite step делает итоговый комментарий короче и понятнее.

LLM-компоненты запускаются локально через LM Studio / Qwen либо через отдельно
настроенный provider. Архитектура при этом остается прежней:
`datasets -> context processing -> models -> validation`.

## 4. Валидация

Оценка строится из нескольких сигналов:

- deterministic similarity;
- `hit@k`;
- `MRR`;
- LLM judge;
- latency/runtime-метрики;
- ручной просмотр примеров;
- в следующих итерациях — CI/test signals.

## MVP

- использовать готовые датасеты;
- реализовать preprocessing и единую MR schema;
- собрать retrieval baseline;
- добавить reranking;
- подключить локальный Qwen pipeline;
- оценить offline;
- запустить demo на одном MR;
- сохранять воспроизводимые run artifacts.

## Локальный слой LLM-экспериментов

После baseline запускаются варианты через LM Studio:

- Qwen generator + baseline logistic reranker;
- baseline retrieval generator + Qwen reranker;
- Qwen generator + Qwen reranker;
- Qwen generator/reranker + rewrite step;
- Qwen generator/reranker + LLM judge.

Для каждого run сохраняются predictions, side-by-side comparison reports,
summary metrics, inference/judge/total wall latency, cached/uncached LLM calls,
token usage, cache hit rate, parse error rate и fallback rate.

## Итерация 2

- улучшить context retrieval;
- провести ручную разметку 20-50 MR;
- сверить LLM judge с человеческой оценкой;
- усилить проверку через CI/test signals;
- исследовать fix suggestions;
- перейти к более реалистичным end-to-end задачам.
