"""Киноманьяк в Telegram: диалог поверх kino.pick. Уровень Legendary, требования в SPEC.md.

    uv run bot.py --provider anthropic     облачная модель, нужен ANTHROPIC_API_KEY
    uv run bot.py --local                  qwen3.5:4b через Ollama, без ключа и бесплатно
    uv run bot.py --provider groq --big    большой каталог: 30 кандидатов отбирает код

Токен бота берётся из TELEGRAM_BOT_TOKEN в .env (создаётся в @BotFather командой /newbot).
Логика диалога живёт в классе Dialog и не зависит от Telegram: её проверяют тесты
с заглушкой вместо модели. Сам Telegram это тонкий слой на long polling Bot API.
"""

import argparse
import contextlib
import difflib
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import OrderedDict

from dotenv import load_dotenv

import kino

HELP = ("Я подбираю фильмы из своего каталога: только реальные фильмы, название, год "
        "и описание из каталога.\n\n"
        "Напишите, что хотите посмотреть: «как Интерстеллар, но не про космос», "
        "«с мамой на казахском, до двух часов».\n"
        "«ещё» — другие фильмы по тому же запросу.\n"
        "«уже смотрел Брат» — больше не предлагать этот фильм.\n"
        "/reset — начать заново.")
MORE = re.compile(r"^\s*(давай\s+)?(ещё|еще|другие|другое|тағы)\W*$", re.IGNORECASE)
# Исключение только в начале сообщения (B11, B11a): внутри обычного запроса
# «…но не советуй ужасы» эти слова остаются частью запроса.
EXCLUDE = re.compile(r"^\s*(?:уже\s+(?:смотрел[аи]?|видел[аи]?)|не\s+предлагай|не\s+советуй)\s+(.+)$"
                     r"|^(.+?)\s+(?:көрдім|көргенмін)\W*$", re.IGNORECASE)
RESET = {"/start", "/reset", "/help"}
TELEGRAM = "https://api.telegram.org/bot{token}/{method}"
MAX_CHATS = 1000   # B16b: сколько диалогов бот держит в памяти


def explain_http(code):
    """Понятная причина остановки для ответов Telegram, которые повтор не исправит (B16a)."""
    return {401: "Telegram не принял токен (401): проверьте TELEGRAM_BOT_TOKEN в .env",
            404: "Telegram не нашёл бота (404): проверьте TELEGRAM_BOT_TOKEN в .env",
            409: "С этим токеном уже работает другой экземпляр бота (409): остановите его "
                 "и запустите бота снова"}.get(code)


class Dialog:
    """Состояние одного чата и ответы на его сообщения (SPEC B5–B15)."""

    def __init__(self, client, model, films, picker=None):
        self.client, self.model, self.films = client, model, films
        self.picker = picker or kino.pick
        self.reset()

    def reset(self):
        self.last_query = None   # запрос, к которому относится «ещё»
        self.shown = set()       # id показанных по этому запросу
        self.excluded = set()    # id «уже смотрел» до конца диалога
        self.pending = None      # запрос, на который бот задал уточняющий вопрос

    def handle(self, text):
        """Ответ бота на одно сообщение пользователя."""
        text = (text or "").strip()
        if not text:
            return HELP
        # В групповом чате команда приходит как «/start@имя_бота» (B14).
        if text.split()[0].split("@")[0].lower() in RESET:
            self.reset()
            return HELP
        if MORE.match(text):                                    # B8, B8a, B9
            if self.pending:
                return "Сначала ответьте на уточняющий вопрос или напишите новый запрос."
            if not self.last_query:
                return "Сначала напишите запрос: какой фильм вы ищете?"
            return self.ask(self.last_query, more=True)
        match = EXCLUDE.search(text)
        name = (match.group(1) or match.group(2)).strip(" «»\"'.,!?") if match else ""
        # С запятой это уже не название, а запрос: «уже видела все комедии, хочу драму» (B11a).
        if match and "," not in name and ";" not in name:      # B11, B12
            return self.exclude(name)
        # Ответ на уточняющий вопрос продолжает исходный запрос (B6).
        query = f"{self.pending}. {text}" if self.pending else text
        self.pending = None
        self.last_query, self.shown = query, set()
        return self.ask(query)

    def available(self):
        """Каталог без показанных и исключённых фильмов: только он уходит модели (B8, B13)."""
        hidden = self.shown | self.excluded
        return [film for film in self.films if str(film["id"]) not in hidden]

    def ask(self, query, more=False):
        films = self.available()
        if not films:                                           # B10
            return "Больше ничего не нашлось. Напишите новый запрос."
        printed = io.StringIO()
        # pick сама печатает фильмы из каталога и сообщения об ошибках: бот отдаёт
        # пользователю ровно этот текст, поэтому показ не расходится с kino.py (B1, B15).
        with contextlib.redirect_stdout(printed):
            if self.picker is kino.pick_big:
                # B16c: кандидаты отбираются по всему каталогу (индекс TF-IDF строится
                # один раз), а показанные и исключённые убираются из кандидатов.
                outcome, ids = kino.pick_big(self.client, self.model, query, self.films,
                                             hidden=self.shown | self.excluded)
            else:
                outcome, ids = self.picker(self.client, self.model, query, films)
        text = printed.getvalue().strip()
        if outcome == "ok":
            self.shown |= {str(i) for i in ids}
            return text + "\n\nНапишите «ещё» для других фильмов или новый запрос."
        if outcome == "clarify":                                # B5, B8a
            self.pending, self.last_query = query, None
            return text
        if outcome == "no_match" and more:                      # B10
            return "Больше ничего не подходит под этот запрос. Напишите новый."
        return text or "Не получилось подобрать фильм. Попробуйте ещё раз."

    def find(self, name):
        """Фильм каталога по названию: точно, по вхождению, затем по похожести."""
        wanted = name.casefold().replace("ё", "е")
        titles = {str(f["title"]).casefold().replace("ё", "е"): f for f in self.films}
        if wanted in titles:
            return titles[wanted]
        inside = [f for t, f in titles.items() if len(wanted) >= 4 and (wanted in t or t in wanted)]
        if len(inside) == 1:
            return inside[0]
        close = difflib.get_close_matches(wanted, list(titles), n=1, cutoff=0.85)
        return titles[close[0]] if close else None

    def exclude(self, name):
        film = self.find(name) if name else None
        if film is None:
            return f"Фильма «{name}» нет в каталоге, исключать нечего."
        self.excluded.add(str(film["id"]))
        self.shown.discard(str(film["id"]))
        return f"Хорошо, «{film['title']}» больше не предложу до конца разговора."


class Chats:
    """Отдельный диалог на каждый чат Telegram (B2)."""

    def __init__(self, client, model, films, picker=None):
        self.client, self.model, self.films, self.picker = client, model, films, picker
        self.dialogs = OrderedDict()   # порядок: от давно неактивных к недавним

    def get(self, chat_id):
        if chat_id not in self.dialogs:
            self.dialogs[chat_id] = Dialog(self.client, self.model, self.films, self.picker)
            while len(self.dialogs) > MAX_CHATS:                # B16b
                self.dialogs.popitem(last=False)
        self.dialogs.move_to_end(chat_id)
        return self.dialogs[chat_id]

    def reply(self, chat_id, text):
        return self.get(chat_id).handle(text)


def telegram(token, method, payload, timeout=40):
    """Вызов Telegram Bot API. Токен не печатается."""
    request = urllib.request.Request(
        TELEGRAM.format(token=token, method=method),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as answer:
        return json.load(answer)


def serve(token, chats):
    """Long polling: забирает сообщения, отвечает каждому чату по очереди."""
    offset = None
    me = telegram(token, "getMe", {}).get("result", {})
    print(f"Бот @{me.get('username', '?')} запущен. Остановить: Ctrl+C", flush=True)
    while True:
        try:
            updates = telegram(token, "getUpdates", {"timeout": 30, "offset": offset,
                                                     "allowed_updates": ["message"]})
        except urllib.error.HTTPError as error:
            # 401, 404, 409 повтор не исправит: бот останавливается с причиной (B16a).
            if explain_http(error.code):
                raise
            print(f"Telegram ответил {error.code}, повтор через 5 с", flush=True)
            time.sleep(5)
            continue
        except (OSError, ValueError) as error:
            print("Telegram недоступен, повтор через 5 с:", type(error).__name__, flush=True)
            time.sleep(5)
            continue
        for update in updates.get("result", []):
            offset = update["update_id"] + 1
            message = update.get("message") or {}
            chat_id, text = (message.get("chat") or {}).get("id"), message.get("text")
            if chat_id is None or not text:
                continue
            print(f"[{chat_id}] {text}", flush=True)
            with contextlib.suppress(OSError, ValueError):
                telegram(token, "sendChatAction", {"chat_id": chat_id, "action": "typing"})
            try:
                reply = chats.reply(chat_id, text)
            except Exception as error:  # noqa: BLE001 бот не падает из-за одного сообщения
                print("Ошибка обработки:", type(error).__name__, error, flush=True)
                reply = "Что-то пошло не так. Попробуйте ещё раз или /reset."
            for start in range(0, len(reply), 4000):            # лимит Telegram 4096 знаков
                try:
                    telegram(token, "sendMessage", {"chat_id": chat_id,
                                                    "text": reply[start:start + 4000]})
                except (OSError, ValueError) as error:
                    print("Не удалось отправить ответ:", type(error).__name__, flush=True)


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Киноманьяк в Telegram")
    kino.add_model_args(parser)
    parser.add_argument("--big", action="store_true",
                        help=f"каталог {kino.BIG_CATALOG}: {kino.K} кандидатов отбирает код")
    args = parser.parse_args(argv)
    load_dotenv()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:                                               # B16
        print("Нет TELEGRAM_BOT_TOKEN в .env: создайте бота в Telegram у @BotFather "
              "(команда /newbot) и впишите токен строкой TELEGRAM_BOT_TOKEN=...")
        return 1
    catalog = kino.BIG_CATALOG if args.big else kino.CATALOG
    films = json.loads(catalog.read_text(encoding="utf-8"))
    client, model = kino.client_from_args(args)
    picker = kino.pick_big if args.big else kino.pick
    print(f"Модель {model}, каталог {catalog} ({len(films)} фильмов)", flush=True)
    try:
        serve(token, Chats(client, model, films, picker))
    except KeyboardInterrupt:
        print("Остановлен")
    except urllib.error.HTTPError as error:
        print(explain_http(error.code) or f"Telegram ответил {error.code}: проверьте TELEGRAM_BOT_TOKEN")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
