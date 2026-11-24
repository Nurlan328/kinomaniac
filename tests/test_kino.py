"""Тесты pick с заглушкой вместо клиента: без сети и без ключа."""

import json
from types import SimpleNamespace

import openai
import pytest
from openai.types import CompletionUsage

import kino

try:
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx

FILMS = [
    {"id": "t-1", "title": "Старик и степь", "year": 2010, "minutes": 95,
     "genres": ["драма"], "countries": ["Казахстан"], "languages": ["казахский"],
     "description": "Чабан в буран спасает отару."},
    {"id": "t-2", "title": "Тёмный квартал", "year": 2018, "minutes": 150,
     "genres": ["боевик"], "countries": ["США"], "languages": ["английский"],
     "description": "Детектив ищет убийцу."},
    {"id": "t-3", "title": "Дом у моря", "year": 2001, "minutes": 110,
     "genres": ["мелодрама"], "countries": ["Франция"], "languages": ["французский"],
     "description": "Лето на побережье."},
    {"id": "t-4", "title": "Ракета", "year": 2020, "minutes": 100,
     "genres": ["фантастика"], "countries": ["США"], "languages": ["английский"],
     "description": "Полёт к Марсу."},
]
REQUEST = httpx.Request("POST", "http://127.0.0.1:9/v1/chat/completions")


def answer(*ids, status="ok", reason=""):
    return {"status": status, "reason": reason,
            "picks": [{"film_id": i, "why": "подходит"} for i in ids]}


class Stub:
    """Заглушка клиента: отдаёт ответ по схеме или бросает ошибку на выборе.

    Вызов со схемой Filters получает filters (по умолчанию пустые условия),
    вызов со схемой Choice получает payload или ошибку, как у настоящего клиента.
    """

    def __init__(self, payload=None, error=None, refusal=None, filters=None):
        self.payload, self.error, self.refusal = payload, error, refusal
        self.filters = filters or {}
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self.parse))

    def parse(self, **kwargs):
        schema = kwargs["response_format"]
        kind = "picks" if "picks" in schema.model_fields else "filters"
        self.calls.append({"kind": kind, **kwargs})
        if kind == "filters":
            return self.reply(schema.model_validate(self.filters))
        if self.error:
            raise self.error
        # Как настоящий parse: ответ не по схеме даёт pydantic.ValidationError.
        parsed = None if self.refusal else schema.model_validate(self.payload)
        return self.reply(parsed, self.refusal)

    def reply(self, parsed, refusal=None):
        message = SimpleNamespace(parsed=parsed, refusal=refusal)
        usage = CompletionUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120)
        return SimpleNamespace(id="chatcmpl-stub", choices=[SimpleNamespace(message=message)],
                               usage=usage)

    def prompt(self, kind="picks"):
        call = next(c for c in self.calls if c["kind"] == kind)
        return "\n".join(m["content"] for m in call["messages"])


@pytest.fixture(autouse=True)
def journal(tmp_path, monkeypatch):
    """Журнал тестов пишется во временную папку, а не в calls.jsonl проекта."""
    path = tmp_path / "calls.jsonl"
    monkeypatch.setattr(kino, "JOURNAL", path)
    return path


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


# ---------- Common ----------

def test_alien_id_is_invalid_and_hidden(journal, capsys):
    outcome, ids = kino.pick(Stub(answer("t-1", "t-999")), "m", "что-нибудь", FILMS)
    assert (outcome, ids) == ("invalid", [])
    out = capsys.readouterr().out
    assert "t-999" not in out and "Старик и степь" not in out
    row = rows(journal)[-1]
    assert row["outcome"] == "invalid" and row["rejected"] == ["t-999"]


def test_repeat_is_shown_once(capsys):
    outcome, ids = kino.pick(Stub(answer("t-2", "t-2")), "m", "боевик", FILMS)
    assert (outcome, ids) == ("ok", ["t-2"])
    assert capsys.readouterr().out.count("Тёмный квартал") == 1


def test_no_match(journal):
    stub = Stub(answer(status="no_match", reason="нет такого"))
    assert kino.pick(stub, "m", "мультфильм 2031 года", FILMS) == ("no_match", [])
    assert rows(journal)[-1]["outcome"] == "no_match"


def test_more_than_three_are_cut_to_three():
    outcome, ids = kino.pick(Stub(answer("t-1", "t-2", "t-3", "t-4")), "m", "всё", FILMS)
    assert outcome == "ok" and ids == ["t-1", "t-2", "t-3"]


def test_ok_prints_title_year_description_from_catalog(capsys, journal):
    outcome, ids = kino.pick(Stub(answer("t-1")), "m", "казахское кино", FILMS)
    assert (outcome, ids) == ("ok", ["t-1"])
    out = capsys.readouterr().out
    assert "Старик и степь" in out and "2010" in out and "Чабан в буран спасает отару." in out
    row = rows(journal)[-1]
    assert row["model"] == "m" and row["outcome"] == "ok" and row["usage"]["prompt_tokens"] == 100


def test_prompt_has_tags_and_whole_catalog():
    stub = Stub(answer("t-1"))
    kino.pick(stub, "m", "кино на вечер", FILMS)
    prompt = stub.prompt("picks")
    assert "<каталог>" in prompt and "<запрос>кино на вечер</запрос>" in prompt
    for film in FILMS:
        assert json.dumps(film["title"], ensure_ascii=False) in prompt
    assert next(c for c in stub.calls if c["kind"] == "picks")["response_format"] is kino.Choice


def test_schema_error_is_invalid(journal):
    stub = Stub({"status": "clarify", "reason": "?", "picks": []})
    assert kino.pick(stub, "m", "что-то", FILMS) == ("invalid", [])
    assert rows(journal)[-1]["outcome"] == "invalid"


def test_refusal():
    assert kino.pick(Stub(refusal="Не могу."), "m", "что-то", FILMS) == ("refusal", [])


def test_no_connection(journal):
    stub = Stub(error=openai.APIConnectionError(request=REQUEST))
    assert kino.pick(stub, "m", "что-то", FILMS) == ("no_connection", [])
    assert rows(journal)[-1]["outcome"] == "no_connection"


def test_api_status_error():
    response = httpx.Response(503, request=REQUEST)
    stub = Stub(error=openai.APIStatusError("Error code: 503", response=response, body=None))
    assert kino.pick(stub, "m", "что-то", FILMS) == ("error_503", [])


def test_pick_uses_passed_catalog_not_films_json():
    other = [{"id": 7, "title": "Проверочный фильм", "year": 1999, "minutes": 90,
              "genres": ["драма"], "countries": ["Казахстан"], "languages": ["казахский"],
              "description": "Описание."}]
    outcome, ids = kino.pick(Stub(answer("7")), "m", "кино", other)
    assert (outcome, ids) == ("ok", [7])
    assert kino.pick(Stub(answer("t-1")), "m", "кино", other) == ("invalid", [])


# ---------- Rare: условия проверяет код ----------

def test_passes_each_condition():
    film = FILMS[1]   # 2018, 150 мин, боевик, английский
    assert kino.passes(film, kino.Filters())
    assert not kino.passes(film, kino.Filters(max_minutes=120))
    assert not kino.passes(film, kino.Filters(year_to=2010))
    assert not kino.passes(film, kino.Filters(year_from=2019))
    assert not kino.passes(film, kino.Filters(languages=["казахский"]))
    assert not kino.passes(film, kino.Filters(exclude_genres=["боевик"]))
    assert not kino.passes(film, kino.Filters(must_genres=["комедия"]))
    assert kino.passes(film, kino.Filters(max_minutes=150, languages=["английский"]))


def test_two_calls_and_journal_steps(journal):
    stub = Stub(answer("t-1"))
    kino.pick(stub, "m", "кино", FILMS)
    assert [c["kind"] for c in stub.calls] == ["filters", "picks"]
    assert [r["step"] for r in rows(journal)] == ["filters", "choice"]
    assert "<запрос>кино</запрос>" in stub.prompt("filters")
    assert '"казахский"' in stub.prompt("filters")   # слова каталога ушли в промпт


def test_violating_film_is_filtered_before_choice_and_not_shown(journal, capsys):
    stub = Stub(answer("t-2", "t-1"), filters={"max_minutes": 100})
    outcome, ids = kino.pick(stub, "m", "до 100 минут", FILMS)
    assert (outcome, ids) == ("ok", ["t-1"])
    assert "Тёмный квартал" not in stub.prompt("picks")     # каталог отфильтрован до выбора
    assert "Тёмный квартал" not in capsys.readouterr().out  # и не показан после ответа
    row = rows(journal)[-1]
    assert row["violations"] == ["t-2"] and row["filters"] == {"max_minutes": 100}


def test_only_violations_is_invalid():
    stub = Stub(answer("t-2"), filters={"max_minutes": 100})
    assert kino.pick(stub, "m", "до 100 минут", FILMS) == ("invalid", [])


def test_empty_candidates_is_no_match_without_second_call(journal):
    # французский фильм в каталоге один и идёт 110 минут: условия несовместимы
    stub = Stub(answer("t-1"), filters={"languages": ["французский"], "max_minutes": 100})
    assert kino.pick(stub, "m", "французское кино до 100 минут", FILMS) == ("no_match", [])
    assert [c["kind"] for c in stub.calls] == ["filters"]
    assert rows(journal)[-1]["step"] == "choice" and rows(journal)[-1]["outcome"] == "no_match"


def test_unknown_filter_words_are_ignored():
    stub = Stub(answer("t-1"), filters={"languages": ["kazakh"], "must_genres": ["family"]})
    assert kino.pick(stub, "m", "кино", FILMS) == ("ok", ["t-1"])
    for film in FILMS:   # условий не осталось, в выбор ушёл весь каталог
        assert json.dumps(film["title"], ensure_ascii=False) in stub.prompt("picks")
