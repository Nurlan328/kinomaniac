"""Прогон набора queries.json: исходы, пойманные чужие id, нарушения условий,
доля пройденных ожиданий.

    uv run check.py                                   облачная модель gpt-6-luna
    uv run check.py --provider groq --pause 45        бесплатный Groq, пауза под лимит токенов
    uv run check.py --provider groq --runs 3          три прогона набора подряд
    uv run check.py --local                           qwen3.5:4b через Ollama
    uv run check.py --big --recall                    Epic: отбор кандидатов без модели
    uv run check.py --big --provider anthropic        Epic: 30 кандидатов, выбор моделью
    uv run check.py --big --whole --provider anthropic  Epic: весь каталог в промпте
    uv run check.py --report [--big]                  таблица сравнения по журналу

Запрос с ожиданием пройден, если исход совпал с ожиданием ("no_match": true
или ok), в ответе есть хотя бы один id из must и нет ни одного из forbid.
Запросы без ожидания только прогоняются: по ним считаются исходы.
"""

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import kino

sys.stdout.reconfigure(encoding="utf-8")

QUERIES = Path("queries.json")
QUERIES_BIG = Path("queries_big.json")


def passed(expect, outcome, ids):
    """True/False по ожиданию, None если ожидания нет."""
    if not expect:
        return None
    if expect.get("no_match"):
        return outcome == "no_match"
    if outcome != "ok":
        return False
    shown = {str(i) for i in ids}
    must = {str(i) for i in expect.get("must", [])}
    forbid = {str(i) for i in expect.get("forbid", [])}
    return (not must or bool(shown & must)) and not (shown & forbid)


def journal_rows():
    if not kino.JOURNAL.exists():
        return []
    return [json.loads(x) for x in
            kino.JOURNAL.read_text(encoding="utf-8").splitlines() if x.strip()]


def run_set(client, model, films, items, pause, picker=kino.pick):
    """Один прогон набора. Возвращает сводку в виде словаря."""
    outcomes, results = Counter(), []
    before = len(journal_rows())
    started = time.time()
    for n, item in enumerate(items, 1):
        if n > 1 and pause:
            time.sleep(pause)
        print(f"\n[{n}/{len(items)}] {item['query']}")
        outcome, ids = picker(client, model, item["query"], films)
        outcomes[outcome] += 1
        verdict = passed(item.get("expect"), outcome, ids)
        results.append((item, outcome, ids, verdict))
        if verdict is False:
            print(f"   НЕ ПРОЙДЕНО: ожидание {item['expect']}, исход {outcome}, показаны {ids}")
        elif verdict:
            print("   пройдено")
    rows = journal_rows()[before:]
    with_expect = [r for r in results if r[3] is not None]
    good = [r for r in with_expect if r[3]]
    return {
        "model": model,
        "outcomes": dict(outcomes),
        "passed": len(good),
        "expected": len(with_expect),
        "rejected": sum(len(r.get("rejected") or []) for r in rows),
        "violations": sum(len(r.get("violations") or []) for r in rows),
        "seconds": sum(r.get("seconds") or 0 for r in rows) / max(len(items), 1),
        "wall": time.time() - started,
        "usd": sum(r.get("usd") or 0 for r in rows),
        "failed": [r for r in with_expect if not r[3]],
    }


def report(summary, title):
    print(f"\n{title}")
    print(f"Модель: {summary['model']}")
    print(f"Исходы: {summary['outcomes']}")
    print(f"Чужих id поймано: {summary['rejected']}; нарушений условий поймано: "
          f"{summary['violations']}")
    share = summary["passed"] / summary["expected"] * 100 if summary["expected"] else 0
    print(f"Пройдено: {summary['passed']} из {summary['expected']} запросов с ожиданием "
          f"({share:.0f}%)")
    print(f"Секунд на запрос (оба вызова): {summary['seconds']:.1f}; "
          f"стоимость прогона: ${summary['usd']:.4f}")
    if summary["failed"]:
        print("Не пройдены:")
        for item, outcome, ids, _ in summary["failed"]:
            print(f"  - {item['query']!r}: исход {outcome}, показаны {ids}")


def seconds_between(earlier, later):
    """Секунды между двумя отметками времени журнала; без отметки бесконечность."""
    try:
        return (datetime.fromisoformat(later) - datetime.fromisoformat(earlier)).total_seconds()
    except (TypeError, ValueError):
        return float("inf")


def report_from_journal(items, mode=None):
    """Таблица сравнения моделей по журналу, без вызовов модели.

    Берутся строки выбора (step choice) с полем filters, то есть прогоны уровня
    Rare. Прогон k у модели это k-е появление каждого запроса набора; считаются
    только полные прогоны, где каждый запрос появился k раз. Это те же числа,
    что печатает check.py по ходу прогона, но их можно получить заново из
    calls.jsonl, например после обрыва по дневному лимиту провайдера.
    mode выбирает строки уровня Epic: candidates или whole; без него только
    строки малого каталога.
    """
    order = [item["query"] for item in items]
    expect = {item["query"]: item.get("expect") for item in items}
    by_model, pending, ignored = {}, {}, Counter()
    for row in sorted(journal_rows(), key=lambda r: r.get("time", "")):
        request = row.get("request") if isinstance(row.get("request"), dict) else {}
        query, model = request.get("query"), row.get("model")
        step = row.get("step", "choice")
        if step == "filters":
            # Ждёт свою строку выбора. Строки фильтров старых прогонов без mode
            # подходят выбору любого режима, новые только своему.
            pending.setdefault((model, query), []).append(row)
            continue
        if step != "choice" or "filters" not in row or request.get("mode") != mode:
            continue
        if query not in expect:
            # Строка этого режима не из набора: ручной запуск или запрос,
            # который с тех пор переформулировали в наборе.
            ignored[model] += 1
            continue
        # Секунды и цена запроса это оба вызова: фильтры и выбор. Строка фильтров
        # берётся последняя неиспользованная для той же модели, запроса и режима,
        # не старше минуты: два прогона одной модели могут идти параллельно,
        # а у раннего no_match в pick_big своего вызова фильтров нет.
        pre = {}
        waiting = pending.get((model, query), [])
        for i in range(len(waiting) - 1, -1, -1):
            candidate = waiting[i]
            if (candidate.get("request") or {}).get("mode") not in (None, mode):
                continue
            if seconds_between(candidate.get("time"), row.get("time")) <= 60:
                pre = waiting.pop(i)
            break
        tokens = sum(((r.get("usage") or {}).get("prompt_tokens") or 0) for r in (row, pre))
        row = dict(row, seconds=(row.get("seconds") or 0) + (pre.get("seconds") or 0),
                   usd=(row.get("usd") or 0) + (pre.get("usd") or 0), tokens=tokens)
        by_model.setdefault(model, []).append(row)
    print("| Модель | Прогонов | Пройдено | Чужих id | Нарушений условий | Вход токенов/запрос "
          "| Сек/запрос | $/прогон |")
    print("|---|---|---|---|---|---|---|---|")
    for model, model_rows in by_model.items():
        # Прогон это проход по запросам в порядке queries.json. Строка не на своём
        # месте (ручной запуск kino.py) пропускается; первый запрос набора после
        # обрыва начинает новый прогон, а оборванный считается неполным.
        runs, current = [], []
        for row in model_rows:
            query = row["request"]["query"]
            if query == order[len(current)]:
                current.append(row)
            elif query == order[0]:
                runs.append(current)
                current = [row]
            else:
                continue
            if len(current) == len(order):
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        complete = [run for run in runs if len(run) == len(order)]
        partial = [len(run) for run in runs if run and len(run) < len(order)]
        if not complete:
            print(f"| `{model}` | 0 полных | — | — | — | — | — | — |"
                  + (f" неполных: {partial} из {len(order)}" if partial else ""))
            continue
        shares, rejected, violations, seconds, usd, tokens = [], 0, 0, [], [], []
        for run in complete:
            good = sum(bool(passed(expect[r["request"]["query"]], r["outcome"],
                                   r.get("shown") or [])) for r in run)
            shares.append(good / len(run))
            rejected += sum(len(r.get("rejected") or []) for r in run)
            violations += sum(len(r.get("violations") or []) for r in run)
            seconds.append(sum(r.get("seconds") or 0 for r in run) / len(run))
            usd.append(sum(r.get("usd") or 0 for r in run))
            tokens.append(sum(r.get("tokens") or 0 for r in run) / len(run))
        n = len(complete)
        print(f"| `{model}` | {n} | "
              f"{sum(shares) / n * 100:.0f}% ({', '.join(f'{s * 100:.0f}%' for s in shares)}) | "
              f"{rejected} | {violations} | {sum(tokens) / n:,.0f} | {sum(seconds) / n:.1f} | "
              f"${sum(usd) / n:.4f} |"
              + (f" неполных: {partial} из {len(order)}" if partial else ""))
    print("\nТокены, секунды и цена на запрос включают оба вызова; цена по прайсу платного доступа.")
    if ignored:
        print("Строк выбора с запросами не из queries.json (не учтены): "
              + ", ".join(f"{m} {n}" for m, n in ignored.items()))


def recall(items, films):
    """Отбор без модели: попал ли фильм из must в кандидаты. Модель не вызывается."""
    titles = {str(f["id"]): f["title"] for f in films}
    started = time.time()
    kino.candidates("прогрев индекса", films)
    print(f"Индекс TF-IDF по {len(films)} фильмам: {time.time() - started:.1f} с")
    hits, total, times = 0, 0, []
    for item in items:
        must = {str(i) for i in (item.get("expect") or {}).get("must", [])}
        started = time.time()
        found = [str(i) for i in kino.candidates(item["query"], films)]
        times.append(time.time() - started)
        if not must:
            continue
        total += 1
        if must & set(found):
            hits += 1
            place = min(found.index(i) for i in must & set(found)) + 1
            print(f"  ok   {item['query']!r}: место {place} из {len(found)}")
        else:
            print(f"  мимо {item['query']!r}: ждали {[titles.get(i, i) for i in sorted(must)][:3]}, "
                  f"первые кандидаты {[titles.get(i, i) for i in found[:3]]}")
    share = hits / total * 100 if total else 0
    print(f"\nОтбор: фильм из must попал в {kino.K} кандидатов в {hits} из {total} запросов "
          f"({share:.0f}%); в среднем {sum(times) / max(len(times), 1) * 1000:.0f} мс на запрос")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Прогон набора запросов")
    kino.add_model_args(parser)
    parser.add_argument("--pause", type=float, default=0,
                        help="секунды между запросами: бесплатный Groq даёт ~8K токенов "
                             "в минуту, а один запрос с каталогом весит до 7K")
    parser.add_argument("--runs", type=int, default=1, help="сколько раз прогнать набор")
    parser.add_argument("--report", action="store_true",
                        help="не вызывать модель: таблица сравнения моделей по calls.jsonl")
    parser.add_argument("--big", action="store_true",
                        help=f"уровень Epic: каталог {kino.BIG_CATALOG} и набор {QUERIES_BIG}")
    parser.add_argument("--whole", action="store_true",
                        help="с --big: весь каталог в промпте вместо 30 кандидатов")
    parser.add_argument("--recall", action="store_true",
                        help="с --big: только отбор кандидатов, без модели")
    args = parser.parse_args()
    if (args.whole or args.recall) and not args.big:
        parser.error("--whole и --recall работают только вместе с --big")
    films = json.loads((kino.BIG_CATALOG if args.big else kino.CATALOG).read_text(encoding="utf-8"))
    items = json.loads((QUERIES_BIG if args.big else QUERIES).read_text(encoding="utf-8"))
    if args.report:
        for mode in (("candidates", "whole") if args.big else (None,)):
            if mode:
                print(f"\nРежим {mode}:")
            report_from_journal(items, mode)
        return 0
    if args.recall:
        return recall(items, films)
    client, model = kino.client_from_args(args)
    if args.big and args.whole:
        def picker(c, m, q, f):
            return kino.pick(c, m, q, f, mode="whole")
    elif args.big:
        picker = kino.pick_big
    else:
        picker = kino.pick

    summaries = []
    for run in range(1, args.runs + 1):
        if run > 1 and args.pause:
            time.sleep(args.pause)
        summary = run_set(client, model, films, items, args.pause, picker)
        summaries.append(summary)
        report(summary, f"Прогон {run} из {args.runs}")
    if args.runs > 1:
        n = len(summaries)
        share = sum(s["passed"] / s["expected"] for s in summaries if s["expected"]) / n * 100
        print(f"\nИтого по {n} прогонам, модель {model}:")
        print(f"  пройдено в среднем {share:.0f}% "
              f"({', '.join(f'{s['passed']}/{s['expected']}' for s in summaries)})")
        print(f"  чужих id до проверки: {sum(s['rejected'] for s in summaries)}; "
              f"нарушений условий: {sum(s['violations'] for s in summaries)}")
        print(f"  секунд на запрос: {sum(s['seconds'] for s in summaries) / n:.1f}; "
              f"стоимость одного прогона: ${sum(s['usd'] for s in summaries) / n:.4f}")
        print(f"  строка для README: | `{model}` | {n} | {share:.0f}% | "
              f"{sum(s['rejected'] for s in summaries)} | "
              f"{sum(s['violations'] for s in summaries)} | "
              f"{sum(s['seconds'] for s in summaries) / n:.1f} | "
              f"${sum(s['usd'] for s in summaries) / n:.4f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
