"""Смысловые регрессии пользовательского журнала, с намеренно ошибающейся LLM."""
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config
from consultant.consultant import EnergytoolConsultant
from consultant.dialogue import customer_facts, human_request
from consultant.evidence import evidence_units
from consultant.answer_policy import safe_quotes, applicable, minimum_data, validate_content
from llm.qwen import LLMError
from integration.interface import EnergytoolAI
from rag.retrieval import Retriever
from rag.build_chunks import load_docs, make_chunks

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = make_chunks(load_docs(ROOT / "knowledge-base"))[0]
BY_ID = {c["id"]: c for c in CHUNKS}


class BadComposer:
    def __init__(self, text="Готово.", flags=None):
        self.text, self.flags, self.messages = text, flags or {}, []

    def chat_json(self, messages):
        self.messages.append(messages)
        data = json.loads(messages[1]["content"])
        if "выбираешь подтверждения" in messages[0]["content"]:
            assert all(u.get("role") != "instruction" for u in data["evidence"])
            return {"intent": "energy_problem", "out_of_scope": False, "wants_human": False,
                    "evidence_ids": [u["id"] for u in data["evidence"]][:8], **self.flags}
        return {"action": "answered", "claims": [{"text": self.text, "evidence_ids": [data["evidence"][0]["id"]]}], "question_slot": ""}


class UserRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = Retriever(kb_path=ROOT / "knowledge-base", mode="bm25")

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        patches = patch.multiple(config, SESSION_DIR=temp.name, GROUNDED_ANSWERS=True, SOURCE_FIRST_RESPONSES=False)
        patches.start()
        self.addCleanup(patches.stop)

    def app(self, llm=None):
        return EnergytoolAI(EnergytoolConsultant(self.retriever, llm or BadComposer()))

    def test_bill_paraphrases_cannot_be_answered_without_verification_step(self):
        for question in ("Перешли на вечернюю смену, стали платить больше. Почему?",
                         "Мы стали больше работать вечером, и счёт вырос. Из-за чего?",
                         "Раз счёт вырос, поставщик точно нас обманывает?"):
            with self.subTest(question=question):
                result = self.app(BadComposer("Почему сосед платит меньше — сравнить нельзя.")).chat(question)
                self.assertEqual(result["status"], "answered")
                self.assertNotIn("сосед", result["answer"])
                self.assertIn("сравнить два периода", result["answer"])
                self.assertIn("средняя цена", result["answer"])
                self.assertIn("плата за мощность", result["answer"])
                self.assertEqual(result["debug"]["question_slot"], "price_category")

    def test_normal_generation_also_gets_required_question(self):
        class Complete(BadComposer):
            def chat_json(self, messages):
                if "выбираешь подтверждения" in messages[0]["content"]:
                    return super().chat_json(messages)
                data = json.loads(messages[1]["content"])
                proof = next(u for u in data["evidence"] if u["kind"] == "guidance")
                scenario = next(u for u in data["evidence"] if u["kind"] == "conditional_scenario")
                return {"action": "answered", "claims": [{"text": "Возможны изменения профиля потребления. Сравните два периода: объём потребления, среднюю цену за кВт·ч и плату за мощность.", "evidence_ids": [proof["id"], scenario["id"]]}], "question_slot": ""}
        app = self.app(Complete())
        result = app.chat("Счёт вырос, стали больше платить вечером. Почему?")
        self.assertEqual(result["debug"]["reason_code"], "grounded_answer")
        self.assertIn("Какая у вас ценовая категория?", result["answer"])
        result2 = app.chat("Счёт всё равно растёт, что проверять?", result["session_id"])
        self.assertNotIn("Какая у вас ценовая категория?", result2["answer"])

    def test_component_growth_cannot_be_contrasted_with_known_bill_growth(self):
        contract = {"task": "bill_diagnosis", "required_evidence_ids": []}
        answer = "Рост может быть связан с пиками. Сравните два периода: объём потребления, среднюю цену за кВт·ч и плату за мощность. В счёте может вырасти только одна составляющая, а не вся сумма."
        with self.assertRaises(LLMError):
            validate_content(answer, {"claims": []}, contract, "Счёт вырос")

    def test_ambiguous_station_needs_type_not_price_category(self):
        model = BadComposer()
        r = self.app(model).chat("У нас нагрузка в основном вечером. Станция точно окупится?")
        self.assertEqual(r["status"], "clarification_needed")
        self.assertIn("типе станции", r["answer"])
        self.assertFalse(model.messages)

    def test_solar_guarantee_requires_calculation_not_evening_diagnosis(self):
        model = BadComposer("Экономика проекта не обеспечивается.")
        r = self.app(model).chat("Солнечная станция с вечерней нагрузкой точно окупится?")
        self.assertEqual(r["debug"]["reason_code"], "source_calculation_limits")
        self.assertIn("расчёт", r["answer"])
        self.assertNotIn("не обеспечивается", r["answer"])
        self.assertNotIn("не совпадает", r["answer"])
        self.assertFalse(model.messages)

    def test_solar_fallback_never_uses_tariff_row_as_primary(self):
        units = evidence_units([BY_ID["price-005"], BY_ID["solar-009"]])
        quotes = safe_quotes(units, [], "Солнечная станция с вечерней нагрузкой точно окупится?", "solar_economics")
        self.assertTrue(quotes)
        self.assertTrue(all(u["product"] == "solar" for u in quotes))

    def test_unrelated_quote_is_not_an_answer_just_because_it_is_first(self):
        units = evidence_units([BY_ID["price-005"]])
        self.assertEqual(safe_quotes(units, [], "Солнечная станция окупится?", "solar_economics"), [])

    def test_expansion_fact_does_not_answer_payback_question(self):
        units = evidence_units([BY_ID["solar-004"]])
        self.assertEqual(safe_quotes(units, [], "Солнечная станция окупится?", "solar_economics"), [])

    def test_old_slate_roof_does_not_establish_private_house(self):
        units = evidence_units([BY_ID["solar-007"]])
        home = next(u for u in units if re.search(r"частн.*жил.*дом", u["quote"], re.I))
        self.assertFalse(applicable(home, "Шиферной крыше 20 лет", "product_question"))
        self.assertTrue(applicable(home, "Частный жилой дом, крыша 20 лет", "product_question"))

    def test_missing_day_night_data_is_not_sufficient_with_long_archive(self):
        r = self.app().chat("Архив потребления за два года, но разбивки день/ночь нет. Этого хватит для поиска потерь?")
        self.assertTrue(r["answer"].startswith("Пока недостаточно"))
        self.assertEqual(r["debug"]["requirement_checks"]["day_night"], "missing")
        self.assertEqual(r["debug"]["requirement_checks"]["archive_depth"], "satisfied")
        self.assertNotIn("load_period", r["debug"]["customer_facts"])

    def test_short_archive_fails_source_minimum_even_with_split(self):
        r = self.app().chat("Для поиска потерь есть разбивка день/ночь, архив за 6 месяцев. Данных хватит?")
        self.assertTrue(r["answer"].startswith("Пока недостаточно"))
        self.assertEqual(r["debug"]["requirement_checks"]["archive_depth"], "missing")

    def test_absent_archive_is_not_a_satisfied_numeric_requirement(self):
        r = self.app().chat("Для поиска потерь есть разбивка день/ночь, но архива нет. Данных хватит?")
        self.assertEqual(r["debug"]["requirement_checks"]["archive_depth"], "missing")
        self.assertTrue(r["answer"].startswith("Пока недостаточно"))

    def test_present_and_unknown_requirements_remain_distinct(self):
        for statement, expected in (("разбивка день/ночь есть", "satisfied"), ("не знаю, есть ли разбивка день/ночь", "unknown")):
            with self.subTest(statement=statement):
                r = self.app().chat("Ищем коммерческие потери. Архив за два года, " + statement + ". Данных хватит?")
                self.assertEqual(r["debug"]["requirement_checks"]["day_night"], expected)
                self.assertNotIn("ночью", r["debug"]["customer_facts"].get("load_period", {}).get("quote", ""))
                if expected == "unknown":
                    self.assertEqual(r["debug"]["question_slot"], "day_night")

    def test_day_night_answer_can_correct_previous_negative(self):
        facts = customer_facts([{"role": "user", "content": "Разбивки день/ночь нет."},
                                {"role": "assistant", "content": "Есть ли отдельные значения за день и ночь?"},
                                {"role": "user", "content": "Да"}])
        self.assertEqual(facts["day_night"]["status"], "satisfied")
        self.assertEqual(facts["day_night"]["turn"], 2)

    def test_greeting_is_not_a_load_profile(self):
        facts = customer_facts([{"role": "user", "content": "Добрый вечер! Сетевая станция работает при отключении сети?"}])
        self.assertNotIn("load_period", facts)

    def test_negative_work_schedule_is_not_positive_night_load(self):
        facts = customer_facts([{"role": "user", "content": "Ночью мы не работаем."}])
        self.assertNotIn("load_period", facts)
        self.assertEqual(facts["excluded_load_period"]["status"], "missing")
        facts = customer_facts([{"role": "user", "content": "Ночью мы не работаем, основная нагрузка днём."}])
        self.assertEqual(facts["load_period"]["value"], "днём")
        self.assertEqual(facts["excluded_load_period"]["value"], "Ночью")

    def test_two_powers_are_bound_to_their_objects(self):
        facts = customer_facts([{"role": "user", "content": "Хотим солнечную станцию 60 кВт. Присоединённая мощность предприятия 900 кВт."}])
        self.assertIn("60", facts["project_power"]["quote"])
        self.assertIn("900", facts["connected_power"]["quote"])
        facts = customer_facts([{"role": "user", "content": "Присоединённая мощность 500 кВт."}])
        self.assertNotIn("project_power", facts)

    def test_direct_handoff_precedes_llm_and_contact_is_saved_locally(self):
        model = BadComposer()
        app = self.app(model)
        r = app.chat("Хочу поговорить со специалистом.")
        self.assertEqual(r["status"], "handoff")
        self.assertFalse(r["debug"]["retrieval_called"])
        self.assertFalse(model.messages)
        r2 = app.chat("Тестовый контакт: test@example.com", r["session_id"])
        state = app._get_session(r2["session_id"])
        self.assertEqual(state.data["lead"]["contacts"], ["test@example.com"])
        self.assertEqual(state.data["lead"]["status"], "saved_locally")

    def test_mentions_and_negations_are_not_contact_requests(self):
        for text in ("Не хочу говорить со специалистом", "Не нужен специалист", "Чем занимается специалист?", "Можно узнать, что делает специалист?"):
            with self.subTest(text=text):
                self.assertFalse(human_request(text))

    def test_cancelled_handoff_does_not_capture_later_contact(self):
        app = self.app()
        r = app.chat("Хочу поговорить со специалистом")
        cancelled = app.chat("Не хочу специалиста. Можно панели на фасаде?", r["session_id"])
        self.assertFalse(app._get_session(cancelled["session_id"]).data["pending_handoff"])
        self.assertIsNone(app._get_session(cancelled["session_id"]).data["lead"])
        self.assertFalse(human_request("Хочу поговорить со специалистом, но не надо связываться со специалистом"))
        self.assertTrue(human_request("Не хочу говорить с оператором, хочу поговорить со специалистом"))

    def test_negative_contact_request_cannot_be_overridden_by_model_flag(self):
        r = self.app(BadComposer(flags={"wants_human": True})).chat("Не хочу специалиста. Можно панели на фасаде?")
        self.assertNotEqual(r["status"], "handoff")

    def test_rules_keep_hierarchy_and_are_not_primary_facts(self):
        units = evidence_units([BY_ID["orem-026"]])
        self.assertTrue(units)
        self.assertTrue(all(u["role"] == "instruction" for u in units))
        self.assertTrue(all(not applicable(u, "Точная цена проекта?", "product_question") for u in units))

    def test_workshop_is_not_restricted_by_retail_example(self):
        r = self.app().chat("В кабинете можно сравнивать расход по цехам?")
        self.assertNotIn("если это однотипные", r["answer"])
        self.assertNotIn("ретейл", r["answer"])
        self.assertTrue(all(u["source_id"] != "portfolio-009" for u in r["debug"]["selected_evidence"]))

    def test_unknown_product_price_needs_product_not_internal_prohibition(self):
        r = self.app().chat("Назовите точную цену проекта для моего предприятия прямо сейчас.")
        self.assertEqual(r["status"], "clarification_needed")
        self.assertIn("какого решения", r["answer"])
        self.assertNotIn("Нижняя граница", r["answer"])
