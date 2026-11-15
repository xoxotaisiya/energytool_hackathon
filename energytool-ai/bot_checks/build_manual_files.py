"""Обновить файлы для ручного копирования из scenarios.json (без модели)."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
cases = json.loads((root / "scenarios.json").read_text(encoding="utf-8"))
names = {"basic": "Основные 20 вопросов", "paraphrases": "Перефразирования",
         "boundaries": "Границы и единицы", "dialogues": "Диалоговые цепочки", "robustness": "Неопределённость и давление"}
md = ["# Вопросы для проверки EnergyTool", "",
      "Отправляйте строки внутри блоков по одной, дожидаясь ответа. Они вводятся",
      "в уже запущенном боте после `Вы:`, а не как команды после `PS E:...>`.",
      "Каждый блок начинается с `/new`. Внутри цепочки сброса нет, кроме явно",
      "вставленной проверки `/new`. Названия, ожидания и запреты боту не отправлять.",
      "Ожидания предназначены для оценки и не являются готовыми ответами.", ""]
plain = ["Вводить в запущенном боте, после Вы:, по одной строке с ожиданием ответа.",
         "Не отправлять подписи сценариев и ожидаемое поведение.", ""]
for group, name in names.items():
    md += ["## " + name, ""]
    for case in [c for c in cases if c["group"] == group]:
        md += ["### " + case["id"] + ". " + case["title"], "", "```text", "/new", *case["turns"], "```", "",
               "Проверить: " + " ".join(case["expected"]), "",
               "Нельзя: " + " ".join(case["forbidden"]), ""]
        plain += ["=" * 70, case["id"] + ". " + case["title"], "СТРОКИ ДЛЯ БОТА:", "/new", *case["turns"],
                  "ПРОВЕРИТЬ (НЕ ОТПРАВЛЯТЬ): " + " ".join(case["expected"]),
                  "НЕЛЬЗЯ (НЕ ОТПРАВЛЯТЬ): " + " ".join(case["forbidden"]), ""]
(root / "QUESTIONS.md").write_text("\n".join(md), encoding="utf-8")
(root / "QUESTIONS.txt").write_text("\n".join(plain), encoding="utf-8-sig")
print(f"Подготовлено {len(cases)} сценариев")
