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
    {"id": "t-2", "title": "Ночной город", "year": 2018, "minutes": 150,
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
    """Заглушка клиента: отдаёт заданный ответ или бросает заданную ошибку."""

    def __init__(self, payload=None, error=None, refusal=None):
        self.payload, self.error, self.refusal = payload, error, refusal
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self.parse))

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        # Как настоящий parse: ответ не по схеме даёт pydantic.ValidationError.
        parsed = None if self.refusal else kwargs["response_format"].model_validate(self.payload)
        message = SimpleNamespace(parsed=parsed, refusal=self.refusal)
        usage = CompletionUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120)
        return SimpleNamespace(id="chatcmpl-stub", choices=[SimpleNamespace(message=message)],
                               usage=usage)


@pytest.fixture(autouse=True)
def journal(tmp_path, monkeypatch):
    """Журнал тестов пишется во временную папку, а не в calls.jsonl проекта."""
    path = tmp_path / "calls.jsonl"
    monkeypatch.setattr(kino, "JOURNAL", path)
    return path


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


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
    assert capsys.readouterr().out.count("Ночной город") == 1


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
    prompt = "\n".join(m["content"] for m in stub.calls[0]["messages"])
    assert "<каталог>" in prompt and "<запрос>кино на вечер</запрос>" in prompt
    for film in FILMS:
        assert json.dumps(film["title"], ensure_ascii=False) in prompt
    assert stub.calls[0]["response_format"] is kino.Choice


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
