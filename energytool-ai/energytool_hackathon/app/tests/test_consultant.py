"""Регрессии прежнего протокола (GROUNDED_ANSWERS=0) для сравнения версий."""
import copy
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config
from consultant.consultant import EnergytoolConsultant
from consultant.contracts import validate_decision
from integration.interface import EnergytoolAI
from llm.qwen import LLMError
from rag.retrieval import Retriever

ROOT = Path(__file__).resolve().parents[1]
DECISION = {"intent": "product_question", "out_of_scope": False, "needs_retrieval": True,
            "wants_human": False, "customer_goal": "", "known_facts": [], "missing_info": [],
            "clarification_question": ""}
FACT = "У нас крыша на два ската — восток и запад. Панели можно поставить на оба?"


class Model:
    def __init__(self, decision=None, output=None):
        self.decision = copy.deepcopy(DECISION if decision is None else decision)
        self.output = output or {"action": "answered", "answer": "Можно использовать оба ската.",
                                 "question": "", "source_ids": ["solar-010"]}
        self.messages = []

    def chat_json(self, messages):
        self.messages.append(messages)
        value = self.decision if len(self.messages) == 1 else self.output
        if isinstance(value, Exception):
            raise value
        return copy.deepcopy(value)


class ConsultantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = Retriever(kb_path=ROOT / "knowledge-base", mode="bm25")

    def setUp(self):
        grounding = patch.object(config, "GROUNDED_ANSWERS", False)
        grounding.start()
        self.addCleanup(grounding.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch.object(config, "SESSION_DIR", self.tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)

    def app(self, model=None, retriever=None):
        model = model or Model()
        return EnergytoolAI(EnergytoolConsultant(
            llm=model, retriever=retriever if retriever is not None else self.retriever)), model

    def test_string_false_is_normalized(self):
        app, model = self.app(Model(dict(DECISION, out_of_scope="false", wants_human="false")))
        r = app.chat(FACT)
        self.assertEqual(r["status"], "answered")
        self.assertFalse(r["debug"]["decision"]["out_of_scope"])
        self.assertIn("normalized_boolean:out_of_scope", r["debug"]["decision_warnings"])

    def test_numbers_are_not_booleans(self):
        with self.assertRaises(LLMError):
            validate_decision(dict(DECISION, out_of_scope=1))

    def test_invalid_intent_types_do_not_raise_unhandled_typeerror(self):
        for intent in ([], {}, None, "invented"):
            with self.subTest(intent=intent), self.assertRaises(LLMError):
                validate_decision(dict(DECISION, intent=intent))

    def test_transport_error_is_not_claimed_to_be_absent_evidence(self):
        app, _ = self.app(Model(LLMError("Соединение не установлено.", "transport")))
        r = app.chat(FACT)
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["debug"]["error_code"], "transport")
        self.assertTrue(r["debug"]["retrieval_called"])
        self.assertFalse(r["debug"]["generation_called"])

    def test_json_syntax_failure_during_analysis_can_still_answer(self):
        app, _ = self.app(Model(LLMError("Некорректный JSON.", "json_format")))
        self.assertEqual(app.chat(FACT)["status"], "answered")

    def test_missing_source_ids_are_not_silent_success(self):
        app, _ = self.app(Model(output={"action": "answered", "answer": "Содержательный ответ", "source_ids": []}))
        self.assertEqual(app.chat(FACT)["status"], "error")

    def test_final_action_wins_over_advisory_flags(self):
        self.assertEqual(EnergytoolAI._get_status(
            {"action": "answered"}, {"out_of_scope": True, "wants_human": True}, "answered"), "answered")

    def test_bad_decision_does_not_block_grounded_generation(self):
        app, _ = self.app(Model(dict(DECISION, out_of_scope="unknown")))
        r = app.chat(FACT)
        self.assertEqual(r["status"], "answered")
        self.assertIn("decision_schema", r["debug"]["decision_warnings"])

    def test_false_scope_classification_cannot_skip_retrieval(self):
        app, model = self.app(Model(dict(DECISION, intent="out_of_scope", out_of_scope=True)))
        r = app.chat(FACT)
        self.assertEqual(r["status"], "answered")
        self.assertTrue(r["debug"]["retrieval_called"])
        self.assertIn("запад-восток используются оба ската", model.messages[-1][1]["content"])

    def test_bm25_none_confidence_is_not_refusal(self):
        app, _ = self.app()
        r = app.chat(FACT)
        self.assertIsNone(r["debug"]["retrieval"]["confidence"])
        self.assertEqual(r["status"], "answered")

    def test_smalltalk_never_calls_model_or_retriever(self):
        app, model = self.app()
        for text in ("Здравствуйте", "Спасибо!", "До свидания"):
            r = app.chat(text)
            self.assertEqual(r["debug"]["action"], "smalltalk")
            self.assertFalse(r["debug"]["retrieval_called"])
        self.assertFalse(model.messages)

    def test_greeting_with_question_is_not_swallowed(self):
        app, model = self.app()
        app.chat("Здравствуйте! " + FACT)
        self.assertEqual(len(model.messages), 2)

    def test_clarification_despite_nonempty_hits_for_advice(self):
        model = Model(dict(DECISION, intent="product_advice", missing_info=["тип объекта"],
                           clarification_question="О каком объекте речь?"),
                      {"action": "clarification", "answer": "О каком объекте речь?",
                       "question": "О каком объекте речь?", "source_ids": []})
        app, _ = self.app(model)
        r = app.chat("Хочу снизить затраты на электроэнергию")
        self.assertTrue(r["debug"]["retrieval"]["chunks"])
        self.assertEqual(r["status"], "clarification_needed")
        self.assertEqual(r["question"], "О каком объекте речь?")
        self.assertEqual(len(model.messages), 2)
        self.assertIn("тип объекта", model.messages[-1][1]["content"])

    def test_fact_question_does_not_require_unnecessary_region(self):
        app, _ = self.app(Model(dict(DECISION, missing_info=["регион"],
                                    clarification_question="Какой регион?")))
        self.assertEqual(app.chat(FACT)["status"], "answered")

    def test_final_clarification_has_explicit_status(self):
        app, _ = self.app(Model(output={"action": "clarification", "answer": "Какой объект?",
                                        "question": "Какой объект?", "source_ids": []}))
        r = app.chat(FACT)
        self.assertEqual(r["status"], "clarification_needed")
        self.assertEqual(r["question"], "Какой объект?")

    def test_followup_question_after_fact_is_visible_in_terminal_answer(self):
        app, _ = self.app(Model(output={"action": "answered", "answer": "Можно использовать оба ската.",
                                        "question": "В каком регионе объект?", "source_ids": ["solar-010"]}))
        r = app.chat(FACT)
        self.assertEqual(r["status"], "answered")
        self.assertIn("В каком регионе объект?", r["answer"])
        self.assertIsNone(r["question"])

    def test_unknown_source_is_technical_error_not_grounded_answer(self):
        app, _ = self.app(Model(output={"action": "answered", "answer": "Цена 1 руб.",
                                        "source_ids": ["invented"]}))
        r = app.chat(FACT)
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["sources"], [])
        self.assertEqual(r["debug"]["generation_attempts"], 2)

    def test_paraphrased_followup_is_not_appended_twice(self):
        app, _ = self.app(Model(output={"action": "answered",
            "answer": "Можно использовать оба ската. Какой у вас регион?",
            "question": "В каком регионе объект?", "source_ids": ["solar-010"]}))
        self.assertEqual(app.chat(FACT)["answer"].count("?"), 1)

    def test_no_evidence_is_distinct_from_api_failure(self):
        for output, status in [
            ({"action": "no_evidence", "answer": "Нет подтверждения, предложу специалиста.",
              "source_ids": []}, "no_evidence"),
            (LLMError("Сервис недоступен.", "transport"), "error"),
        ]:
            app, _ = self.app(Model(output=output))
            self.assertEqual(app.chat(FACT)["status"], status)

    def test_handoff_uses_validated_boolean(self):
        app, _ = self.app(Model(dict(DECISION, wants_human=True, out_of_scope=True)))
        r = app.chat("Хочу связаться с менеджером")
        self.assertEqual(r["status"], "handoff")
        self.assertEqual(r["debug"]["reason_code"], "user_requested_human")

    def test_session_continues_and_new_session_is_empty(self):
        app, _ = self.app()
        first = app.chat("Здравствуйте")
        second = app.chat("Спасибо", session_id=first["session_id"])
        self.assertEqual(first["session_id"], second["session_id"])
        state = app._get_session(first["session_id"])
        self.assertEqual(len(state.history()), 4)
        third = app.chat("Здравствуйте")
        self.assertNotEqual(third["session_id"], first["session_id"])
        self.assertEqual(len(app._get_session(third["session_id"]).history()), 2)

    def test_context_uses_more_than_five_hits_when_budget_allows(self):
        hits = self.retriever.retrieve(FACT, top_k=10)
        context, ids = EnergytoolConsultant._context(hits)
        self.assertGreater(len(set(ids).intersection(c["id"] for c in hits["chunks"])), 5)

    def test_all_demo_reach_generation_with_valid_decisions(self):
        with (ROOT / "demo_questions.csv").open(encoding="utf-8-sig", newline="") as f:
            questions = list(csv.DictReader(f))
        class EchoEvidence(Model):
            def chat_json(self, messages):
                self.messages.append(messages)
                if len(self.messages) == 1:
                    return copy.deepcopy(DECISION)
                import re
                ids = re.findall(r"\[([^|\]\n]+) \|", messages[-1]["content"])
                return {"action": "answered", "answer": "Контролируемый ответ теста.",
                        "source_ids": [ids[0].strip()]}
        for q in questions:
            with self.subTest(id=q["id"]):
                app, _ = self.app(EchoEvidence())
                r = app.chat(q["question"])
                self.assertEqual(r["status"], "answered")
                self.assertTrue(r["debug"]["generation_called"])


if __name__ == "__main__":
    unittest.main()
