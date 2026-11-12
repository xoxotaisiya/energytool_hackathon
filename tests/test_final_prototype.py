"""Tests for the final web prototype and lead workflow."""

import tempfile
import unittest
from pathlib import Path

import config
from consultant.consultant import EnergytoolConsultant
from memory.conversation_state import ConversationState


class LeadWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.previous_session_dir = config.SESSION_DIR
        config.SESSION_DIR = self.temp_dir.name
        self.bot = EnergytoolConsultant(retriever=object(), llm=object())
        self.state = ConversationState()

    def tearDown(self):
        config.SESSION_DIR = self.previous_session_dir
        self.temp_dir.cleanup()

    def test_lead_is_collected_and_summarised(self):
        self.state.add("user", "Нужно контролировать расход на нескольких объектах")
        self.state.add("assistant", "Уточните, пожалуйста, задачу.")
        first = self.bot.reply("Хочу связаться с менеджером", self.state)
        self.assertEqual(first["debug"]["stage"], "lead_collecting")

        self.bot.reply("Иванов Иван Иванович", self.state)
        self.bot.reply("8 (999) 123-45-67", self.state)
        result = self.bot.reply("ООО Тест Энерго", self.state)

        lead = result["state"].data["lead"]
        self.assertEqual(result["debug"]["stage"], "lead_complete")
        self.assertEqual(lead["status"], "complete")
        self.assertEqual(lead["phone"], "+79991234567")
        self.assertIn("нескольких объектах", lead["summary"])
        self.assertTrue(Path(self.temp_dir.name, f"{self.state.session_id}.json").exists())

    def test_invalid_phone_is_requested_again(self):
        self.bot.reply("Нужен менеджер", self.state)
        self.bot.reply("Иванов Иван Иванович", self.state)
        result = self.bot.reply("123", self.state)
        self.assertIn("Не удалось распознать", result["answer"])
        self.assertEqual(self.state.data["lead"]["pending_field"], "phone")

    def test_greeting_does_not_call_llm(self):
        result = self.bot.reply("Привет", self.state)
        self.assertEqual(result["debug"]["stage"], "greeting")
        self.assertIn("Расскажите", result["answer"])

    def test_regular_answer_uses_one_model_call(self):
        class FakeRetriever:
            def retrieve(self, question, history=None):
                return {
                    "chunks": [{"id": "kb-1", "path": "О продукте", "text": "Подтверждённый факт"}],
                    "guards": [],
                    "mode": "bm25",
                    "no_answer": False,
                }

        class FakeLLM:
            calls = 0

            def chat_json(self, messages):
                self.calls += 1
                return {"answer": "Ответ по базе знаний.", "status": "answered"}

        llm = FakeLLM()
        bot = EnergytoolConsultant(retriever=FakeRetriever(), llm=llm)
        result = bot.reply("Расскажите о продукте", self.state)
        self.assertEqual(llm.calls, 1)
        self.assertEqual(result["debug"]["stage"], "rag_answer")
        self.assertEqual(result["debug"]["retrieval"]["chunks"][0]["id"], "kb-1")


class WebApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.previous_session_dir = config.SESSION_DIR
        config.SESSION_DIR = self.temp_dir.name

    def tearDown(self):
        config.SESSION_DIR = self.previous_session_dir
        self.temp_dir.cleanup()

    def test_health_and_session_endpoints(self):
        from fastapi.testclient import TestClient
        from web.app import app

        with TestClient(app) as client:
            self.assertEqual(client.get("/api/health").json()["status"], "ok")
            created = client.post("/api/sessions").json()
            session_id = created["session_id"]
            loaded = client.get(f"/api/sessions/{session_id}")
            self.assertEqual(loaded.status_code, 200)
            self.assertEqual(loaded.json()["session_id"], session_id)
            self.assertEqual(len(client.get("/api/sessions").json()["sessions"]), 1)


if __name__ == "__main__":
    unittest.main()
