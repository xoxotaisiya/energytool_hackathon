"""Проверка экспортера диалогов без сети и настоящей модели."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config
from integration.interface import EnergytoolAI
from llm.qwen import LLMError
from tools.run_acceptance import execute_cases, load_cases


class RecordingConsultant:
    def __init__(self):
        self.received = []

    def reply(self, message, state):
        before = state.history()
        self.received.append((message, before))
        state.add("user", message)
        if message == "error":
            raise LLMError("Sensitive service text must not be exported", "transport")
        answer = "previous_user_turns=" + str(sum(m["role"] == "user" for m in before))
        debug = {"action": "answered", "reason_code": "test", "retrieval": {}, "source_ids": []}
        state.add("assistant", answer, debug)
        return {"answer": answer, "state": state, "debug": debug}


def case(cid, turns):
    return {"id": cid, "title": cid, "group": "dialogues", "turns": turns,
            "expected": ["NEVER_SEND_EXPECTATIONS_TO_MODEL"], "forbidden": ["NEVER_SEND_FORBIDDEN_TO_MODEL"]}


class AcceptanceExportTests(unittest.TestCase):
    def test_actual_replies_are_kept_in_history_and_sessions_are_isolated(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            with patch.object(config, "SESSION_DIR", str(output / "sessions")):
                bot = RecordingConsultant()
                report = {"cases": []}
                execute_cases(EnergytoolAI(bot), [case("a", ["first", "second", "/new", "third"]), case("b", ["fourth"])], 1, report, output)
            saved = json.loads((output / "results.json").read_text(encoding="utf-8"))
            first, second = saved["cases"]
            self.assertEqual(first["turns"][1]["answer"], "previous_user_turns=1")
            self.assertEqual(first["turns"][3]["answer"], "previous_user_turns=0")
            self.assertEqual(second["turns"][0]["answer"], "previous_user_turns=0")
            self.assertNotEqual(first["turns"][0]["session_id"], first["turns"][2]["session_id"])
            self.assertNotEqual(first["turns"][3]["session_id"], second["turns"][0]["session_id"])
            self.assertEqual(bot.received[1][1][-1]["content"], "previous_user_turns=0")
            self.assertNotIn("NEVER_SEND", json.dumps(bot.received))
            self.assertIn("previous_user_turns=1", (output / "results.txt").read_text(encoding="utf-8"))
            self.assertIn("NEVER_SEND_EXPECTATIONS", (output / "results.txt").read_text(encoding="utf-8"))

    def test_technical_error_keeps_prior_results_and_sanitizes_exception(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            with patch.object(config, "SESSION_DIR", str(output / "sessions")):
                report = {"cases": []}
                execute_cases(EnergytoolAI(RecordingConsultant()), [case("a", ["ok", "error"])], 1, report, output)
            saved = (output / "results.json").read_text(encoding="utf-8")
            turns = json.loads(saved)["cases"][0]["turns"]
            self.assertEqual(len(turns), 2)
            self.assertEqual(turns[0]["status"], "answered")
            self.assertEqual(turns[1]["status"], "error")
            self.assertEqual(turns[1]["debug"]["error_code"], "transport")
            self.assertNotIn("Sensitive service text", saved)

    def test_repeated_transport_failures_stop_after_saving_partial_log(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            with patch.object(config, "SESSION_DIR", str(output / "sessions")):
                report = {"cases": []}
                with self.assertRaises(LLMError):
                    execute_cases(EnergytoolAI(RecordingConsultant()), [case("a", ["error"] * 5)], 1, report, output)
            saved = json.loads((output / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(len(saved["cases"][0]["turns"]), 3)
            self.assertFalse(saved["cases"][0]["completed"])

    def test_catalog_has_all_groups_and_valid_filters(self):
        catalog = config.ROOT.parent.parent / "bot_checks/scenarios.json"
        self.assertEqual(len(load_cases(catalog)), 49)
        self.assertEqual(len(load_cases(catalog, "basic")), 20)
        self.assertEqual(len(load_cases(catalog, "dialogues")), 7)
        self.assertEqual([c["id"] for c in load_cases(catalog, "all", ["D02"])], ["D02"])
        with self.assertRaises(ValueError):
            load_cases(catalog, "all", ["absent"])
