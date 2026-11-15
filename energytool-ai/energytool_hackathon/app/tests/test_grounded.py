"""Регрессии обнаруженных ошибок консультации: реальные цитаты, управляемая LLM."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config
from consultant.consultant import EnergytoolConsultant
from consultant.dialogue import customer_facts, allowed_questions
from consultant.evidence import evidence_units, grounded_claims, bounded_evidence, prompt_evidence, contextual_constraints
from integration.interface import EnergytoolAI
from llm.qwen import LLMError
from rag.build_chunks import load_docs, make_chunks
from rag.retrieval import Retriever

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = make_chunks(load_docs(ROOT / "knowledge-base"))[0]
BY_ID = {c["id"]: c for c in CHUNKS}


class GroundModel:
    def __init__(self, choose=lambda u: "Ориентация скатов" in u["quote"], text="При ориентации восток–запад можно использовать оба ската.",
                 intent="product_question", slot="", flags=None):
        self.choose, self.text, self.intent, self.slot = choose, text, intent, slot
        self.flags = flags or {}
        self.messages = []

    def chat_json(self, messages):
        self.messages.append(copy.deepcopy(messages))
        data = json.loads(messages[1]["content"])
        if "выбираешь подтверждения" in messages[0]["content"]:
            return dict({"intent": self.intent, "out_of_scope": False, "wants_human": False,
                         "evidence_ids": [u["id"] for u in data["evidence"] if self.choose(u)][:8],
                         "question_slot": self.slot}, **self.flags)
        return {"action": "answered", "claims": [{"text": self.text, "evidence_ids": [data["evidence"][0]["id"]]}],
                "question_slot": self.slot}


class GroundedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = Retriever(kb_path=ROOT / "knowledge-base", mode="bm25")

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        # Здесь проверяется именно генератор и его fallback; source-first покрыт отдельно.
        patcher = patch.multiple(config, SESSION_DIR=tmp.name, GROUNDED_ANSWERS=True, SOURCE_FIRST_RESPONSES=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def app(self, model):
        return EnergytoolAI(EnergytoolConsultant(self.retriever, model))

    def test_bill_growth_fallback_keeps_possible_causes_guidance_and_unknown_category(self):
        class BillModel(GroundModel):
            def chat_json(self, messages):
                if "выбираешь подтверждения" in messages[0]["content"]:
                    data = json.loads(messages[1]["content"])
                    units = data["evidence"]
                    ids = [u["id"] for term in ("ваш профиль", "плата за мощность", "нерегулируемая цена")
                           for u in units if u.get("kind") == "conditional_scenario" and term in u["quote"]]
                    return {"intent": "energy_problem", "out_of_scope": False, "wants_human": False, "evidence_ids": ids}
                raise LLMError("Неверный пересказ", "answer_schema")
        r = self.app(BillModel()).chat("Здравствуйте! Мы стали больше работать вечером, и счёт вырос. Можно разобраться, из-за чего?")
        self.assertEqual(r["debug"]["reason_code"], "extractive_fallback")
        self.assertIn("Возможный сценарий", r["answer"])
        self.assertIn("дорогие часы", r["answer"])
        self.assertIn("из-за пиков", r["answer"])
        self.assertIn("сравнить два периода", r["answer"])
        self.assertIn("Какая у вас ценовая категория?", r["answer"])
        again = self.app(BillModel()).chat("А как проверить рост счёта?", r["session_id"])
        self.assertNotIn("Какая у вас ценовая категория?", again["answer"])
        known = self.app(BillModel()).chat("У нас четвёртая ценовая категория. Стали больше работать вечером, и счёт вырос. Из-за чего?")
        self.assertEqual(known["debug"]["reason_code"], "extractive_fallback")
        self.assertNotIn("Какая у вас ценовая категория?", known["answer"])

    def test_default_pipeline_answers_with_real_quote_and_normalized_false(self):
        m = GroundModel(flags={"out_of_scope": "false", "wants_human": "false"})
        r = self.app(m).chat("У нас крыша на два ската — восток и запад. Панели можно поставить на оба?")
        self.assertEqual(r["status"], "answered")
        self.assertEqual(r["sources"][0]["id"], "solar-010")
        self.assertTrue(r["debug"]["retrieval_called"])
        self.assertEqual(len(m.messages), 2)
        self.assertEqual(r["debug"]["selected_evidence"][0]["source_id"], "solar-010")

    def test_generation_does_not_receive_unselected_properties(self):
        m = GroundModel()
        self.app(m).chat("У нас крыша восток и запад. Можно поставить панели на оба ската?")
        final = json.loads(m.messages[1][1]["content"])
        self.assertTrue(final["evidence"])
        self.assertTrue(all("Ориентация скатов" in u["quote"] for u in final["evidence"]))
        self.assertNotIn("Grade A", json.dumps(final["evidence"], ensure_ascii=False))

    def test_known_orientation_cannot_be_asked_again(self):
        m = GroundModel(intent="product_advice", slot="roof_orientation")
        r = self.app(m).chat("У нас крыша восток и запад. Можно поставить панели на оба ската?")
        self.assertEqual(r["status"], "answered")
        self.assertNotIn("?", r["answer"])
        self.assertNotIn("roof_orientation", r["debug"]["allowed_question_slots"])

    def test_customer_information_comes_only_from_user(self):
        facts = customer_facts([{"role": "assistant", "content": "У вас завод в Москве, 50 счётчиков."},
                                {"role": "user", "content": "У нас три склада и сорок счётчиков. Кровля восток и запад."}])
        self.assertIn("склад", facts["object_type"]["quote"])
        self.assertIn("сорок", facts["meter_count"]["quote"])
        self.assertNotIn("region", facts)
        self.assertTrue(all(f["turn"] == 1 for f in facts.values()))

    def test_project_power_does_not_become_connected_power(self):
        facts = customer_facts([{"role": "user", "content": "Хотим солнечную станцию 60 кВт."}])
        self.assertIn("project_power", facts)
        self.assertNotIn("connected_power", facts)

    def test_two_years_does_not_imply_day_night_data(self):
        history = [{"role": "user", "content": "По одним приборам почасовые, по другим только суточные. Архив за два года."}]
        facts = customer_facts(history)
        self.assertIn("archive_depth", facts)
        self.assertNotIn("day_night", facts)
        units = evidence_units([BY_ID["calc-007"]])
        self.assertIn("day_night", allowed_questions(units, facts, []))

    def test_store_accounting_cannot_use_wholesale_restriction(self):
        units = evidence_units([BY_ID["orem-007"], BY_ID["askue-009"]], ["askue"])
        self.assertTrue(units)
        self.assertFalse(any(u["product"] == "orem" for u in units))

    def test_workshop_accounting_cannot_use_price_category_material(self):
        units = evidence_units([BY_ID["price-013"], BY_ID["askue-004"]], ["askue"])
        self.assertTrue(units)
        self.assertFalse(any(u["source_id"] == "price-013" for u in units))
        ranked = bounded_evidence(units, "Хочу видеть расход электричества отдельно по цехам", 5000)
        self.assertIn("Учёт по цехам", ranked[0]["quote"])

    def test_grid_station_question_cannot_use_isolated_site_economics(self):
        units = evidence_units([BY_ID["solar-018"], BY_ID["solar-008"]], ["solar"],
                               "Обычная сетевая станция с вечерней нагрузкой имеет смысл?")
        self.assertTrue(units)
        self.assertFalse(any(u["source_id"] == "solar-018" for u in units))

    def test_late_load_keeps_applicability_constraint(self):
        facts = customer_facts([{"role": "user", "content": "Больше всего потребляем вечером."}])
        constraints = contextual_constraints(evidence_units([BY_ID["solar-007"]]), facts)
        self.assertEqual(len(constraints), 1)
        self.assertIn("выработка не совпадёт", constraints[0]["quote"])

    def test_estimate_inputs_are_ranked_above_billing_limit(self):
        units = evidence_units([BY_ID["askue-012"], BY_ID["calc-006"]], ["askue"])
        ranked = bounded_evidence(units, "Какие данные понадобятся для оценки платформы учёта?", 4000)
        self.assertEqual(ranked[0]["source_id"], "calc-006")

    def test_quotes_remain_inside_source_and_table_row_keeps_both_columns(self):
        units = evidence_units([BY_ID["solar-008"], BY_ID["solar-007"]])
        for u in units:
            self.assertIn(u["quote"], " ".join(BY_ID[u["source_id"]]["text"].split()))
        off = next(u for u in units if "При отключении внешней сети" in u["quote"])
        self.assertIn("Сетевая станция:", off["quote"])
        self.assertIn("Гибридная станция:", off["quote"])
        slate = next(u for u in units if "шифера" in u["quote"])
        self.assertIn("Технически возможно, экономически сомнительно", slate["quote"])

    def test_context_budget_keeps_whole_quotes(self):
        original = evidence_units([BY_ID["solar-008"], BY_ID["solar-010"]])
        selected = bounded_evidence(original, "оба ската восток запад", 1000)
        self.assertTrue(selected)
        self.assertLessEqual(len(json.dumps(prompt_evidence(selected), ensure_ascii=False)), 1000)
        self.assertTrue(all(u["quote"] in {s["quote"] for s in original} for u in selected))

    def test_hallucinated_number_is_rejected(self):
        unit = evidence_units([BY_ID["solar-010"]])[0]
        with self.assertRaises(LLMError):
            grounded_claims({"claims": [{"text": "Цена 999 рублей.", "evidence_ids": [unit["id"]]}]}, [unit], {}, {unit["source_id"]})

    def test_wrong_power_in_time_example_is_rejected(self):
        unit = {"id": "F1", "source_id": "solar-032", "quote": "Срок реализации типовой станции 100 кВт — около 2 месяцев."}
        facts = {"project_power": {"quote": "60 кВт"}}
        for text in ("Станция 60 кВт будет готова за 2 месяца.", "Срок реализации — 2 месяца."):
            with self.subTest(text=text), self.assertRaises(LLMError):
                grounded_claims({"claims": [{"text": text, "evidence_ids": ["F1"]}]}, [unit], facts, {"solar-032"})

    def test_example_and_smaller_client_power_in_same_claim_do_not_imply_less_time(self):
        unit = {"id": "F1", "source_id": "solar-029", "quote": "Для типовой станции 100 кВт — около 2 месяцев."}
        with self.assertRaises(LLMError):
            grounded_claims({"claims": [{"text": "Типовая станция 100 кВт устанавливается за 2 месяца, а для 60 кВт срок будет меньше.",
                                         "evidence_ids": ["F1"]}]}, [unit], {"project_power": {"quote": "60 кВт"}}, {"solar-029"})

    def test_power_number_cannot_become_price_or_energy(self):
        unit = {"id": "F1", "source_id": "solar-004", "quote": "Мощность станции 500 кВт."}
        for text in ("Стоимость 500 рублей.", "Потребление 500 кВт·ч."):
            with self.subTest(text=text), self.assertRaises(LLMError):
                grounded_claims({"claims": [{"text": text, "evidence_ids": ["F1"]}]}, [unit], {}, {"solar-004"})

    def test_unasked_project_duration_is_not_sent_for_expansion_question(self):
        units = evidence_units([BY_ID["solar-029"], BY_ID["solar-004"]], ["solar"],
                               "Можно сначала поставить солнечную станцию на 60 кВт, а потом расширить?")
        self.assertFalse(any(u["source_id"] == "solar-029" for u in units))
        self.assertTrue(any("наращивается" in u["quote"] for u in units))

    def test_source_scenario_is_not_a_diagnosis(self):
        unit = {"id": "F1", "source_id": "tariffs-001", "kind": "conditional_scenario",
                "quote": "Что выросло: плата за мощность; Как это выглядит: выросла из-за пиков."}
        with self.assertRaises(LLMError):
            grounded_claims({"claims": [{"text": "У вас счёт вырос из-за платы за мощность.", "evidence_ids": ["F1"]}]},
                            [unit], {}, {"tariffs-001"})

    def test_example_in_next_sentence_does_not_excuse_categorical_diagnosis(self):
        unit = {"id": "F1", "source_id": "tariffs-001", "kind": "conditional_scenario", "quote": "Плата за мощность может вырасти из-за пиков."}
        with self.assertRaises(LLMError):
            grounded_claims({"claims": [{"text": "Счёт растёт из-за пиков нагрузки вечером. Для экономии можно, например, перенести нагрузку.",
                                         "evidence_ids": ["F1"]}]}, [unit], {}, {"tariffs-001"})

    def test_evening_load_does_not_establish_night_work(self):
        unit = next(u for u in evidence_units([BY_ID["solar-007"]]) if "ночью" in u["quote"])
        with self.assertRaises(LLMError):
            grounded_claims({"claims": [{"text": "Станция с вечерней нагрузкой не подходит, так как объект работает преимущественно ночью.",
                                         "evidence_ids": [unit["id"]]}]}, [unit], {"load_period": {"quote": "вечером"}}, {"solar-007"})

    def test_night_condition_does_not_establish_insufficient_evening_generation(self):
        unit = next(u for u in evidence_units([BY_ID["solar-007"]]) if "ночью" in u["quote"])
        customer = {"load_period": {"quote": "работаем в основном вечером"}}
        for text in ("Если объект работает ночью, станция не подходит. При вечерней нагрузке выработка будет недостаточной.",
                     "Если объект работает ночью, станция не подходит. При вечерней нагрузке и ночном производстве эффективность снижается.",
                     "Для вечерней нагрузки выработка не совпадает с нагрузкой, поэтому окупаемость не может быть гарантирована.",
                     "Для объекта с вечерней нагрузкой выработка не совпадает с нагрузкой, поэтому экономика проекта не обеспечивается. Окупаемость не может быть гарантирована без расчёта."):
            with self.subTest(text=text), self.assertRaises(LLMError):
                grounded_claims({"claims": [{"text": text, "evidence_ids": [unit["id"]]}]}, [unit], customer, {"solar-007"})
        answer, _ = grounded_claims({"claims": [{"text": "Если объект работает ночью, выработки может быть недостаточно; для вечерней нагрузки нужен отдельный расчёт.",
                                                "evidence_ids": [unit["id"]]}]}, [unit], customer, {"solar-007"})
        self.assertIn("отдельный расчёт", answer)

    def test_failed_paraphrase_then_model_refusal_preserves_real_evidence(self):
        class RefusesRepair(GroundModel):
            def chat_json(self, messages):
                if len(self.messages) == 2:
                    self.messages.append(copy.deepcopy(messages))
                    return {"action": "no_evidence", "claims": [], "question_slot": ""}
                return super().chat_json(messages)
        r = self.app(RefusesRepair(text="Монтаж стоит 99999 рублей.")).chat("У нас крыша восток и запад. Панели можно поставить на оба ската?")
        self.assertEqual(r["status"], "answered")
        self.assertEqual(r["debug"]["reason_code"], "extractive_fallback")
        self.assertNotIn("99999", r["answer"])
        self.assertIn("восток", r["answer"].lower())

    def test_start_question_falls_back_to_source_inputs_instead_of_product_pitch(self):
        class Fixed:
            def retrieve(self, *args, **kwargs):
                return {"chunks": [BY_ID["calc-006"]], "guards": [], "mode": "bm25"}
        model = GroundModel(choose=lambda u: True, text="Рекомендуем использовать облачную платформу учёта.")
        api = EnergytoolAI(EnergytoolConsultant(Fixed(), model))
        r = api.chat("Хотим подключить платформу учёта в бизнес-центре. С чего начать?")
        self.assertEqual(r["status"], "answered")
        self.assertIn("количество и тип приборов", r["answer"])
        self.assertIn("схема сбора данных", r["answer"])
        self.assertIn("задача учёта", r["answer"])
        self.assertEqual(r["debug"]["reason_code"], "source_inputs_answer")

    def test_inputs_question_adds_source_requirements_for_active_product(self):
        result = self.retriever.retrieve("Хотим подключить платформу учёта в бизнес-центре. Что понадобится для оценки?")
        all_hits = result["chunks"] + result["supporting_chunks"]
        self.assertTrue(any(c["product"] == "askue" and c["section_type"] == "inputs-needed" for c in all_hits))

    def test_current_photo_collection_does_not_switch_platform_product(self):
        model = GroundModel(choose=lambda u: True)
        api = self.app(model)
        from memory.conversation_state import ConversationState
        state = ConversationState()
        state.add("user", "Хотим подключить к платформе приборы в бизнес-центре, учитывать расходы по арендаторам.")
        state.add("assistant", "Как сейчас собираете показания?")
        r = api.chat("Кто-то присылает фото, по некоторым ходит техник. С чего начать?", state.session_id)
        self.assertEqual(r["debug"]["reason_code"], "source_inputs_answer")
        self.assertIn("количество и тип приборов", r["answer"])
        self.assertNotIn("архива фотографий", r["answer"])
        self.assertNotIn("meter-reading", r["debug"]["active_products"])
        self.assertTrue(all(u["product"] != "meter-reading" for u in r["debug"]["selected_evidence"]))

    def test_unknown_day_night_condition_is_asked_even_if_model_omits_question(self):
        class Fixed:
            def retrieve(self, *args, **kwargs):
                return {"chunks": [BY_ID["calc-007"]], "guards": [], "mode": "bm25"}
        model = GroundModel(choose=lambda u: "Минимально достаточно" in u["quote"],
                            text="Минимум — суточные данные с разбивкой день и ночь и архив не менее года.")
        api = EnergytoolAI(EnergytoolConsultant(Fixed(), model))
        r = api.chat("Мы электросетевая компания. Архив за два года. Таких данных хватит, чтобы начать?")
        self.assertEqual(r["status"], "answered")
        self.assertIn("отдельные значения потребления за день и ночь", r["answer"])
        self.assertEqual(r["debug"]["question_slot"], "day_night")

    def test_unsupported_id_and_internal_id_are_rejected(self):
        unit = evidence_units([BY_ID["solar-010"]])[0]
        for text, ids in (("Согласно F1, монтаж возможен.", [unit["id"]]), ("Монтаж возможен.", ["invented"])):
            with self.subTest(text=text), self.assertRaises(LLMError):
                grounded_claims({"claims": [{"text": text, "evidence_ids": ids}]}, [unit], {}, {unit["source_id"]})

    def test_asked_slot_survives_session_and_new_session_resets_it(self):
        m = GroundModel(choose=lambda u: "шифера" in u["quote"], text="На старой шиферной кровле монтаж экономически сомнителен.",
                        intent="product_advice", slot="roof_age")
        api = self.app(m)
        first = api.chat("У нас старая шиферная крыша. Стоит рассмотреть солнечные панели?")
        self.assertIn("Сколько лет кровле?", first["answer"])
        second = api.chat("Пока не знаю возраст, продолжим.", first["session_id"])
        self.assertNotIn("Сколько лет кровле?", second["answer"])
        self.assertIn("roof_age", api._get_session(first["session_id"]).data["asked_slots"])
        fresh = api.chat("Здравствуйте")
        self.assertNotIn("asked_slots", api._get_session(fresh["session_id"]).data)

    def test_model_transport_error_stays_error(self):
        class Offline:
            def chat_json(self, messages):
                raise LLMError("Недоступно", "transport")
        r = self.app(Offline()).chat("Можно панели на оба ската?")
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["debug"]["error_code"], "transport")

    def test_bad_numeric_paraphrase_falls_back_to_selected_quote(self):
        r = self.app(GroundModel(text="Монтаж стоит 999 рублей.")).chat(
            "У нас крыша восток и запад. Можно поставить панели на оба ската?")
        self.assertEqual(r["status"], "answered")
        self.assertNotIn("999", r["answer"])
        self.assertIn("оба ската", r["answer"])
        self.assertEqual(r["debug"]["reason_code"], "extractive_fallback")
        self.assertEqual(r["sources"][0]["id"], "solar-010")
        self.assertTrue(r["debug"]["answer_validation_errors"])

    def test_sim_alias_is_search_only_and_not_an_added_answer(self):
        from rag.query_normalization import normalize_query
        q = "В счётчиках симки другого оператора"
        self.assertIn("сим-карты", normalize_query(q))
        self.assertNotIn("менять", normalize_query(q))
        result = self.retriever.retrieve(q)
        self.assertIn("симки", result["original_query"])
        self.assertTrue(any("сим-карт" in c["text"] for c in result["chunks"]))


if __name__ == "__main__":
    unittest.main()
