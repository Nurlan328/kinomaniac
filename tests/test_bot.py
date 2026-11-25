"""Тесты уровня Legendary по требованиям SPEC.md (B5–B17). Без сети и ключа.

Пока bot.py нет, весь файл пропускается (pytest.importorskip): тесты
закоммичены раньше кода, и хук pre-commit при этом не падает.
"""

import json

import openai
import pytest

import kino
from test_kino import FILMS, REQUEST, Stub, answer, rows

bot = pytest.importorskip("bot")
check = pytest.importorskip("check")


class SeqStub(Stub):
    """Заглушка с очередью ответов: каждый вызов выбора берёт следующий ответ."""

    def __init__(self, *payloads, filters=None):
        super().__init__(payload=None, filters=filters)
        self.queue = list(payloads)

    def parse(self, **kwargs):
        if "picks" in kwargs["response_format"].model_fields:
            item = self.queue.pop(0)
            if isinstance(item, Exception):
                self.calls.append({"kind": "picks", **kwargs})
                raise item
            self.payload = item
        return super().parse(**kwargs)

    def picks_prompts(self):
        """Каталог и запрос из сообщения пользователя: системный промпт с примером
        («Старик и степь» в примере совпадает с тестовым фильмом) не берётся."""
        return [c["messages"][-1]["content"] for c in self.calls if c["kind"] == "picks"]


@pytest.fixture(autouse=True)
def journal(tmp_path, monkeypatch):
    path = tmp_path / "calls.jsonl"
    monkeypatch.setattr(kino, "JOURNAL", path)
    return path


def dialog(*payloads):
    stub = SeqStub(*payloads)
    return bot.Dialog(stub, "m", FILMS), stub


def title_in(film_id, text):
    return next(f["title"] for f in FILMS if f["id"] == film_id) in text


# ---------- B5–B7: уточнение ----------

def test_b7_pick_returns_clarify_without_films(journal, capsys):
    stub = Stub(answer(status="clarify", reason="Какой жанр вам ближе?"))
    assert kino.pick(stub, "m", "посоветуй фильм", FILMS) == ("clarify", [])
    assert "Какой жанр вам ближе?" in capsys.readouterr().out
    assert rows(journal)[-1]["outcome"] == "clarify"


def test_b5_b6_clarify_question_then_combined_query():
    d, stub = dialog(answer(status="clarify", reason="Какой жанр?"), answer("t-1"))
    first = d.handle("посоветуй что-нибудь")
    assert "Какой жанр?" in first and not any(title_in(f["id"], first) for f in FILMS)
    second = d.handle("драму")
    assert title_in("t-1", second)
    prompt = stub.picks_prompts()[-1]
    assert "посоветуй что-нибудь" in prompt and "драму" in prompt


# ---------- B8–B10: «ещё» ----------

def test_b8_more_gives_new_films_without_repeats():
    d, stub = dialog(answer("t-1"), answer("t-3"))
    assert title_in("t-1", d.handle("спокойное кино"))
    reply = d.handle("ещё")
    assert title_in("t-3", reply) and not title_in("t-1", reply)
    assert json.dumps("Старик и степь", ensure_ascii=False) not in stub.picks_prompts()[-1]
    assert "спокойное кино" in stub.picks_prompts()[-1]


def test_b9_more_before_any_query_does_not_call_model():
    d, stub = dialog()
    reply = d.handle("еще")
    assert "запрос" in reply.lower() and stub.calls == []


def test_b10_more_when_everything_shown():
    d, _ = dialog(answer("t-1", "t-2", "t-3"), answer("t-4"),
                  answer(status="no_match", reason="больше нет"))
    d.handle("всё подряд")
    d.handle("ещё")
    reply = d.handle("ещё")
    assert "больше" in reply.lower() or "ничего" in reply.lower()


# ---------- B11–B13: исключения ----------

def test_b11_b13_excluded_film_never_reaches_model():
    d, stub = dialog(answer("t-1"), answer("t-3"))
    reply = d.handle("уже смотрел Тёмный квартал")
    assert "Тёмный квартал" in reply
    d.handle("что-нибудь")
    d.handle("ещё")
    for prompt in stub.picks_prompts():
        assert json.dumps("Тёмный квартал", ensure_ascii=False) not in prompt


def test_b11_kazakh_exclusion_form():
    d, _ = dialog()
    d.handle("Дом у моря көрдім")
    assert "t-3" in d.excluded


def test_b12_unknown_title_excludes_nothing():
    d, stub = dialog()
    reply = d.handle("уже смотрел Аватар 7")
    assert "нет" in reply.lower() and not d.excluded and stub.calls == []


# ---------- B14–B16: сброс и ошибки ----------

def test_b14_reset_clears_dialog():
    d, _ = dialog(answer("t-1"))
    d.handle("уже смотрел Дом у моря")
    d.handle("кино")
    d.handle("/reset")
    assert not d.excluded and not d.shown and d.last_query is None


def test_b15_model_error_is_reported_and_dialog_continues():
    d, _ = dialog(openai.APIConnectionError(request=REQUEST), answer("t-1"))
    assert "связи" in d.handle("кино на вечер").lower()
    assert title_in("t-1", d.handle("кино на вечер"))


def test_b16_no_token_exits_without_network(monkeypatch, capsys):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setattr(bot, "load_dotenv", lambda *a, **k: None, raising=False)
    assert bot.main(["--provider", "ollama"]) == 1
    assert "TELEGRAM_BOT_TOKEN" in capsys.readouterr().out


def test_b2_dialogs_are_separate_per_chat():
    stub = SeqStub(answer("t-1"))
    chats = bot.Chats(stub, "m", FILMS)
    chats.reply(1, "уже смотрел Дом у моря")
    assert "t-3" in chats.get(1).excluded and not chats.get(2).excluded


# ---------- B17: надёжность ----------

def test_b17_stability_share():
    expect = {"a": {"must": ["t-1"]}, "b": {"no_match": True}}
    run = lambda a_ok: [{"request": {"query": "a"}, "outcome": "ok", "shown": ["t-1"] if a_ok else []},  # noqa: E731
                        {"request": {"query": "b"}, "outcome": "no_match"}]
    single, stable = check.stability([run(True), run(False), run(True)], expect)
    assert single == pytest.approx(5 / 6) and stable == pytest.approx(1 / 2)
