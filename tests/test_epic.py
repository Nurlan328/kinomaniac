"""Тесты уровня Epic: отбор кандидатов без модели и pick_big с заглушкой."""

import json

import pytest

import kino
from test_kino import Stub, answer, rows

BIG = [
    {"id": f"q{n}", "title": f"Фильм {n}", "year": 1960 + n % 60, "minutes": 80 + n % 90,
     "genres": ["драма"], "countries": ["США"], "languages": ["английский"],
     "description": f"Обычная история номер {n} о людях и городе."}
    for n in range(1, 1101)
] + [
    {"id": "shark", "title": "Зубы моря", "year": 1975, "minutes": 124,
     "genres": ["ужасы", "приключения"], "countries": ["США"], "languages": ["английский"],
     "description": "Огромная белая акула нападает на курортный городок, шериф выходит в море."},
    {"id": "steppe", "title": "Буран в степи", "year": 2012, "minutes": 102,
     "genres": ["драма", "приключения"], "countries": ["Казахстан"],
     "languages": ["казахский", "русский"],
     "description": "Старый чабан в зимней казахской степи спасает отару от волков."},
    {"id": "batyr", "title": "Тысяча батыров", "year": 2012, "minutes": 132,
     "genres": ["исторический", "военный"], "countries": ["Казахстан"], "languages": ["казахский"],
     "description": "XVIII век, юный джигит собирает отряд и защищает степь от врагов."},
]


@pytest.fixture(autouse=True)
def journal(tmp_path, monkeypatch):
    path = tmp_path / "calls.jsonl"
    monkeypatch.setattr(kino, "JOURNAL", path)
    return path


def test_candidates_find_by_description_words():
    found = kino.candidates("фильм ужасов про акулу", BIG)
    assert found[0] == "shark" and len(found) <= kino.K


def test_candidates_apply_rule_filters():
    found = kino.candidates("что посмотреть на казахском, до двух часов", BIG)
    assert found == ["steppe"]                       # 132 минуты не проходит
    found = kino.candidates("қазақша тарихи фильм", BIG)
    assert found[0] == "batyr"


def test_candidates_without_matches_keep_catalog_order():
    found = kino.candidates("абракадабра", BIG)
    assert found == [f["id"] for f in BIG[:kino.K]]


def test_rule_filters_negation_and_duration():
    f = kino.rule_filters("старое кино до 100 минут, кроме комедий", BIG + [
        dict(BIG[0], id="c", genres=["комедия"])])
    assert f.year_to == 1991 and f.max_minutes == 100 and f.exclude_genres == ["комедия"]


def test_pick_big_sends_only_candidates(journal):
    stub = Stub(answer("shark"))
    outcome, ids = kino.pick_big(stub, "m", "фильм ужасов про акулу", BIG)
    assert (outcome, ids) == ("ok", ["shark"])
    prompt = stub.prompt("picks")
    assert len(prompt) < len(json.dumps(BIG, ensure_ascii=False)) * 0.2
    assert json.dumps("Фильм 1100", ensure_ascii=False) not in prompt
    assert rows(journal)[-1]["request"]["mode"] == "candidates"


def test_pick_big_hides_alien_id():
    assert kino.pick_big(Stub(answer("shark", "q-alien")), "m", "акула", BIG) == ("invalid", [])


def test_title_word_is_not_a_reference():
    films = BIG[-3:] + [dict(BIG[0], id="love", title="Любовь", description="Пожилая пара в Париже.")]
    assert "love" in kino.candidates("грустный фильм про любовь", films)
    assert "love" not in kino.candidates("что-то как «Любовь»", films)


def test_named_genre_ranks_but_does_not_cut():
    films = BIG[-3:] + [dict(BIG[0], id="pianist", title="Пианист", genres=["драма"],
                             description="Музыкант выживает в гетто.")]
    assert "pianist" in kino.candidates("кино про музыканта", films)


def test_kazakh_adverb_and_short_decade():
    assert kino.rule_filters("қазақша драма", BIG).languages == ["казахский"]
    f = kino.rule_filters("боевики 90-х", BIG + [dict(BIG[0], id="x", genres=["боевик"])])
    assert (f.year_from, f.year_to) == (1990, 1999)


def test_ordinary_words_are_not_year_conditions():
    for query in ("фильм про старшеклассников", "кино про старика в степи", "новогодний фильм"):
        f = kino.rule_filters(query, BIG)
        assert f.year_from is None and f.year_to is None, query
    assert kino.rule_filters("старое доброе кино", BIG).year_to == 1991


def test_negation_runs_through_a_list():
    films = BIG + [dict(BIG[0], id="h", genres=["ужасы"]), dict(BIG[0], id="t", genres=["триллер"])]
    f = kino.rule_filters("без ужасов и триллеров", films)
    assert f.exclude_genres == ["ужасы", "триллер"] and not f.must_genres


def test_pick_big_keeps_candidate_order():
    stub = Stub(answer("shark"))
    kino.pick_big(stub, "m", "фильм ужасов про акулу", BIG)
    prompt = stub.prompt("picks")
    assert prompt.index("Зубы моря") < prompt.index("Фильм 1\"")


def test_short_title_needs_capital_letter():
    films = BIG[-3:] + [dict(BIG[0], id="she", title="Она", description="Писатель и ИИ.")]
    assert "she" in kino.candidates("фильм, как она любит, про любовь", films)
    assert "she" not in kino.candidates("что-то как Она", films)
