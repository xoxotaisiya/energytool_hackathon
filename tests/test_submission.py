"""Проверки обязательного формата автоматического сабмита."""

import csv
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from build_submission import build_archive
from solution import SubmissionBot, run_submission, validate_submission


class FakeRetriever:
    def retrieve(self, question):
        return {
            "no_answer": False,
            "chunks": [
                {
                    "id": "test-001",
                    "path": "Тестовый продукт › Назначение",
                    "text": "Подтверждённый фрагмент базы знаний.",
                }
            ],
            "guards": [],
        }


class FakeLLM:
    def __init__(self):
        self.messages = None

    def chat(self, messages):
        self.messages = messages
        return "Краткий подтверждённый ответ."


class FakeBot:
    def answer(self, question):
        return "Ответ на: " + question


class SubmissionTests(unittest.TestCase):
    def test_bot_passes_question_and_context_to_single_llm_call(self):
        llm = FakeLLM()
        bot = SubmissionBot(retriever=FakeRetriever(), llm=llm)
        result = bot.answer("Как работает продукт?")
        self.assertEqual(result, "Краткий подтверждённый ответ.")
        prompt = llm.messages[-1]["content"]
        self.assertIn("Как работает продукт?", prompt)
        self.assertIn("Подтверждённый фрагмент", prompt)

    def test_questions_are_converted_to_valid_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            questions = root / "questions.csv"
            output = root / "submission.csv"
            questions.write_text(
                'id,question\n1,"Первый вопрос"\n2,"Вопрос, с запятой"\n',
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"MAX_WORKERS": "1"}):
                run_submission(questions, output, bot=FakeBot())
            validate_submission(output, ["1", "2"])
            with output.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["id"] for row in rows], ["1", "2"])
            self.assertTrue(all(row["answer"] for row in rows))

    def test_builder_creates_flat_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = build_archive(Path(tmp) / "submission.zip")
            with zipfile.ZipFile(archive_path) as archive:
                names = archive.namelist()
            self.assertIn("solution.py", names)
            self.assertIn("requirements.txt", names)
            self.assertTrue(all("/" not in name for name in names))


if __name__ == "__main__":
    unittest.main()
