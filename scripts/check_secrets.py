"""Проверка ключей API в файлах коммита. Запускается хуком pre-commit.

    uv run python scripts/check_secrets.py ФАЙЛ...

Шаблоны те же, что ищет автопроверка курса (ключи OpenAI, Anthropic, Google, Groq),
плюс токен Telegram-бота. Файл .env в коммит не допускается вообще.
"""

import re
import sys
from pathlib import Path

KEY = re.compile(r"\b(sk-(?:proj-|svcacct-|ant-)?[A-Za-z0-9_\-]{20,}"
                 r"|AIza[0-9A-Za-z_\-]{35}|gsk_[A-Za-z0-9]{40,}"
                 r"|\d{8,10}:[A-Za-z0-9_\-]{35})")   # последний шаблон: токен Telegram-бота


def main(paths):
    problems = []
    for name in paths:
        path = Path(name)
        if path.name == ".env":
            problems.append(f"{name}: файл .env не коммитится")
            continue
        if not path.is_file() or path.stat().st_size > 5_000_000:
            continue
        for number, line in enumerate(path.read_text("utf-8", errors="ignore").splitlines(), 1):
            if KEY.search(line):
                problems.append(f"{name}:{number}: похоже на ключ API")
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
