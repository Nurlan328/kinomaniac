"""Журнал calls.jsonl: исход, токены, секунды и цена вызовов.

    uv run show.py        последние 10 вызовов и итог по всему журналу
    uv run show.py all    все вызовы
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

JOURNAL = Path("calls.jsonl")
LAST = 10


def reasoning(usage):
    details = usage.get("completion_tokens_details") or {}
    return details.get("reasoning_tokens") or 0


def table(rows):
    print(f"{'время':19}  {'модель':12}  {'шаг':7}  {'исход':14}  {'вход':>6}  "
          f"{'выход':>5}  {'сек':>5}  {'$':>8}")
    for row in rows:
        usage = row.get("usage") or {}
        print(f"{row.get('time', ''):19}  {str(row.get('model', '')):12}  "
              f"{row.get('step', 'choice'):7}  {row.get('outcome', ''):14}  "
              f"{usage.get('prompt_tokens', 0):>6}  {usage.get('completion_tokens', 0):>5}  "
              f"{row.get('seconds', 0):>5}  {row.get('usd', 0) or 0:>8.5f}")


def summary(rows):
    print()
    print(f"Вызовов: {len(rows)}")
    by_model = defaultdict(list)
    for row in rows:
        by_model[row.get("model")].append(row)
    for model, items in by_model.items():
        outcomes = Counter(r.get("outcome") for r in items)
        usages = [r.get("usage") or {} for r in items]
        total_in = sum(u.get("prompt_tokens", 0) for u in usages)
        total_out = sum(u.get("completion_tokens", 0) for u in usages)
        total_reasoning = sum(reasoning(u) for u in usages)
        usd = sum(r.get("usd") or 0 for r in items)
        rejected = sum(len(r.get("rejected") or []) for r in items)
        priced = [r["usd"] for r in items if r.get("usd")]
        steps = Counter(r.get("step", "choice") for r in items)
        violations = sum(len(r.get("violations") or []) for r in items)
        print(f"\n{model}: {len(items)} вызовов ("
              + ", ".join(f"{k} {v}" for k, v in outcomes.most_common()) + "); по шагам: "
              + ", ".join(f"{k} {v}" for k, v in steps.most_common()))
        print(f"  токены: вход {total_in}, выход {total_out}, "
              f"из них рассуждения {total_reasoning}")
        print(f"  отклонённых чужих id: {rejected}; фильмов не по условиям: {violations}")
        print(f"  стоимость: ${usd:.4f} всего"
              + (f", ${sum(priced) / len(priced):.5f} за вызов в среднем, "
                 f"10 000 запросов ≈ ${sum(priced) / len(priced) * 10_000:.2f}"
                 if priced else ""))


def main():
    if not JOURNAL.exists():
        print("Журнал пуст: файла calls.jsonl ещё нет")
        return
    lines = JOURNAL.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    table(rows if "all" in sys.argv[1:] else rows[-LAST:])
    summary(rows)


if __name__ == "__main__":
    main()
