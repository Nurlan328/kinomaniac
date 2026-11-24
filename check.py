"""Прогон набора queries.json: исходы, пойманные чужие id, пройденные ожидания.

    uv run check.py                  облачная модель gpt-6-luna
    uv run check.py --provider groq --pause 60   бесплатный Groq, пауза под лимит токенов в минуту
    uv run check.py --local          локальная qwen3.5:4b через Ollama

Запрос с ожиданием пройден, если исход совпал с ожиданием ("no_match": true
или ok), в ответе есть хотя бы один id из must и нет ни одного из forbid.
Запросы без ожидания только прогоняются: по ним считаются исходы.
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import kino

sys.stdout.reconfigure(encoding="utf-8")

QUERIES = Path("queries.json")


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


def main():
    parser = argparse.ArgumentParser(description="Прогон набора запросов")
    kino.add_model_args(parser)
    parser.add_argument("--pause", type=float, default=0,
                        help="секунды между запросами: бесплатный Groq даёт ~8K токенов "
                             "в минуту, а один вызов с каталогом весит почти столько же")
    args = parser.parse_args()
    films = json.loads(kino.CATALOG.read_text(encoding="utf-8"))
    items = json.loads(QUERIES.read_text(encoding="utf-8"))
    client, model = kino.client_from_args(args)
    before = len(journal_rows())

    outcomes = Counter()
    results = []
    for n, item in enumerate(items, 1):
        if n > 1 and args.pause:
            time.sleep(args.pause)
        print(f"\n[{n}/{len(items)}] {item['query']}")
        outcome, ids = kino.pick(client, model, item["query"], films)
        outcomes[outcome] += 1
        verdict = passed(item.get("expect"), outcome, ids)
        results.append((item, outcome, ids, verdict))
        if verdict is False:
            print(f"   НЕ ПРОЙДЕНО: ожидание {item['expect']}, исход {outcome}, показаны {ids}")
        elif verdict:
            print("   пройдено")

    rows = journal_rows()[before:]
    rejected = sum(len(r.get("rejected") or []) for r in rows)
    usd = sum(r.get("usd") or 0 for r in rows)
    with_expect = [r for r in results if r[3] is not None]
    good = [r for r in with_expect if r[3]]
    print(f"\nМодель: {model}")
    print(f"Исходы: {dict(outcomes)}")
    print(f"Чужих id поймано: {rejected}")
    print(f"Пройдено: {len(good)} из {len(with_expect)} запросов с ожиданием")
    print(f"Стоимость прогона: ${usd:.4f}")
    failed = [r for r in with_expect if not r[3]]
    if failed:
        print("Не пройдены:")
        for item, outcome, ids, _ in failed:
            print(f"  - {item['query']!r}: исход {outcome}, показаны {ids}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
