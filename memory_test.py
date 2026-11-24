"""Опыт с памятью модели: год и длительность 10 фильмов без каталога.

    uv run memory_test.py            облачная модель
    uv run memory_test.py --local    qwen3.5:4b через Ollama

Модель получает только названия и отвечает по памяти, код сравнивает ответ
с films.json и считает расхождения. Вызов записывается в calls.jsonl
со step "memory".
"""

import argparse
import json
import sys
import time

import openai
import pydantic
from pydantic import BaseModel

import kino

sys.stdout.reconfigure(encoding="utf-8")

# Пять известных фильмов и пять казахстанских: по ним модель ошибается чаще.
TITLES = ["f-01", "f-39", "f-41", "f-49", "f-52", "kz-01", "kz-02", "kz-03", "kz-04", "kz-05"]
PROMPT = ("Для каждого фильма из списка назови год выхода и длительность в минутах "
          "по памяти, без оговорок. Отвечай только JSON по схеме.")


class Fact(BaseModel):
    title: str
    year: int
    minutes: int


class Facts(BaseModel):
    films: list[Fact]


def main():
    parser = argparse.ArgumentParser()
    kino.add_model_args(parser)
    args = parser.parse_args()
    films = {f["id"]: f for f in json.loads(kino.CATALOG.read_text(encoding="utf-8"))}
    chosen = [films[i] for i in TITLES]
    titles = [f"{f['title']} ({', '.join(f['countries'])})" for f in chosen]
    client, model = kino.client_from_args(args)
    request = {"query": "год и длительность без каталога", "titles": titles}
    started = time.time()
    try:
        completion = client.chat.completions.parse(
            model=model,
            messages=[{"role": "system", "content": PROMPT},
                      {"role": "user", "content": "\n".join(titles)}],
            response_format=Facts,
            max_completion_tokens=kino.LIMIT,
            reasoning_effort=kino.EFFORT.get(model, "none"),
        )
    except (openai.OpenAIError, pydantic.ValidationError) as error:
        print("Вызов не удался:", error)
        kino.log(model, request, "error_unknown", started, step="memory")
        return 1
    answer = completion.choices[0].message.parsed
    kino.log(model, request, "ok", started, completion, step="memory")

    by_title = {f.title.lower(): f for f in answer.films}
    mismatches = 0
    print(f"{'фильм':40}  {'каталог':>12}  {'модель':>12}")
    for film in chosen:
        fact = by_title.get(film["title"].lower())
        if fact is None:
            fact = next((f for f in answer.films if film["title"].lower() in f.title.lower()
                         or f.title.lower() in film["title"].lower()), None)
        got = f"{fact.year}/{fact.minutes}" if fact else "нет ответа"
        wrong = fact is None or fact.year != film["year"] or fact.minutes != film["minutes"]
        mismatches += wrong
        mark = "  <- расхождение" if wrong else ""
        print(f"{film['title'][:40]:40}  {film['year']}/{film['minutes']:<7}  {got:>12}{mark}")
    print(f"\nМодель: {model}. Расхождений с каталогом: {mismatches} из {len(chosen)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
