"""Граница HTTP/JSON/консультант/публичный chat без внешнего сервиса."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import httpx
import config
from consultant.consultant import EnergytoolConsultant
from integration.interface import EnergytoolAI
from llm.qwen import LLMClient
from rag.retrieval import Retriever


class WireIntegrationTests(unittest.TestCase):
    def test_string_flags_travel_through_real_json_client_without_refusal(self):
        responses = [
            {"intent": "product_question", "out_of_scope": "false", "wants_human": "false",
             "needs_retrieval": True, "evidence_ids": ["F1"]},
            {"action": "answered", "claims": [{"text": "При ориентации восток–запад можно использовать оба ската.",
                                                "evidence_ids": ["F1"]}]},
        ]
        payloads = []
        def handler(request):
            payloads.append(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": json.dumps(responses[len(payloads)-1], ensure_ascii=False)},
                                             "done_reason": "stop"})
        llm = LLMClient(httpx.Client(transport=httpx.MockTransport(handler)))
        self.addCleanup(llm.close)
        r = Retriever(kb_path=Path(__file__).resolve().parents[1] / "knowledge-base", mode="bm25")
        with tempfile.TemporaryDirectory() as tmp:
            with patch.multiple(config, SESSION_DIR=tmp, LLM_PROVIDER="ollama", OLLAMA_THINK=False, GROUNDED_ANSWERS=True):
                result = EnergytoolAI(EnergytoolConsultant(r, llm)).chat(
                    "У нас крыша на два ската — восток и запад. Панели можно поставить на оба?")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["sources"][0]["id"], "solar-010")
        self.assertEqual(len(payloads), 2)
        self.assertFalse(payloads[0]["think"])
        self.assertIn("num_predict", payloads[0]["options"])
        self.assertIsInstance(payloads[0]["format"], dict)
        self.assertIn("evidence_ids", payloads[0]["format"]["required"])
        self.assertIn("claims", payloads[1]["format"]["required"])

    def test_context_budget_preserves_whole_chunks(self):
        result = {"chunks": [{"id": str(i), "path": "section", "text": "x" * 200} for i in range(30)], "guards": []}
        with patch.object(config, "CONTEXT_MAX_CHARS", 1000):
            context, ids = EnergytoolConsultant._context(result)
        self.assertLessEqual(len(context), 1000)
        self.assertEqual(context.count("x"), 200 * len(ids))
