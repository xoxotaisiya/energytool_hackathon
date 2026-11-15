import json
import os
import unittest
from unittest.mock import patch
import httpx
import config
from llm.qwen import LLMClient, LLMError


class LLMTests(unittest.TestCase):
    def client(self, handler):
        c = LLMClient(httpx.Client(transport=httpx.MockTransport(handler)))
        self.addCleanup(c.close)
        return c

    def test_openai_payload_and_json(self):
        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(body["max_tokens"], 777)
            self.assertEqual(body["response_format"], {"type": "json_object"})
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}]})
        with patch.multiple(config, LLM_PROVIDER="openai", LLM_BASE_URL="https://model.invalid/v1", MAX_TOKENS=777):
            self.assertTrue(self.client(handler).chat_json([])["ok"])

    def test_platform_does_not_require_ollama_json_schema_support(self):
        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(body["response_format"], {"type": "json_object"})
            self.assertNotIn("format", body)
            self.assertNotIn("think", body)
            self.assertEqual(body["model"], "provided-platform-model")
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}]})
        with patch.multiple(config, LLM_PROVIDER="openai", LLM_MODEL="provided-platform-model", LLM_BASE_URL="https://model.invalid/v1"):
            self.assertTrue(self.client(handler).chat_structured([], {"type": "object"})["ok"])

    def test_malformed_api_and_token_limit_are_typed(self):
        bodies = [
            ({"choices": []}, "response_format"),
            ({"choices": [{"message": None}]}, "response_format"),
            ({"choices": [{"message": {"content": ""}}]}, "empty_content"),
            ({"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]}, "token_limit"),
        ]
        with patch.multiple(config, LLM_PROVIDER="openai", LLM_BASE_URL="https://model.invalid"):
            for body, code in bodies:
                with self.subTest(code=code), self.assertRaises(LLMError) as raised:
                    self.client(lambda req: httpx.Response(200, json=body)).chat([])
                self.assertEqual(raised.exception.code, code)

    def test_json_syntax_and_object_checked(self):
        with patch.object(LLMClient, "chat", return_value='["not object"]'):
            with self.assertRaises(LLMError) as raised:
                self.client(lambda req: None).chat_json([])
        self.assertEqual(raised.exception.code, "json_format")

    def test_exact_ollama_tag_is_required(self):
        with patch.multiple(config, LLM_PROVIDER="ollama", OLLAMA_MODEL="qwen3:8b"):
            client = self.client(lambda req: httpx.Response(200, json={"models": [{"name": "qwen3:4b"}]}))
            with self.assertRaises(LLMError) as raised:
                client.check_connection()
            self.assertEqual(raised.exception.code, "model_missing")

    def test_thinking_only_model_rejects_disabled_thinking_at_startup(self):
        def handler(request):
            if request.url.path.endswith('/api/tags'):
                return httpx.Response(200, json={"models": [{"name": "qwen3:4b"}]})
            return httpx.Response(200, json={"thinking": {"values": [True], "default": True}})
        with patch.multiple(config, LLM_PROVIDER="ollama", OLLAMA_MODEL="qwen3:4b", OLLAMA_THINK=False):
            with self.assertRaises(LLMError) as raised:
                self.client(handler).check_connection()
            self.assertEqual(raised.exception.code, "thinking_mode")

    def test_thinking_trace_is_not_returned_as_content(self):
        with patch.multiple(config, LLM_PROVIDER="ollama", OLLAMA_THINK=True):
            result = self.client(lambda request: httpx.Response(200, json={
                "message": {"thinking": "internal trace", "content": '{"ok":true}'}, "done_reason": "stop"})).chat_json([])
            self.assertEqual(result, {"ok": True})

    def test_native_timing_metrics_are_isolated_between_workers(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        barrier = Barrier(2)
        with patch.multiple(config, LLM_PROVIDER="ollama"):
            client = self.client(lambda request: httpx.Response(200, json={
                "message": {"content": "ok"}, "load_duration": 2000000, "eval_count": 3}))
            def request_one(_):
                client.reset_metrics()
                client.chat([])
                barrier.wait(timeout=5)
                return client.metrics()
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(request_one, range(2)))
            self.assertTrue(all(len(items) == 1 for items in results))
            self.assertTrue(all(items[0]["load_duration_ms"] == 2 for items in results))

    def test_temporary_failure_retried_but_auth_not_retried(self):
        calls = []
        def handler(req):
            calls.append(req)
            if len(calls) == 1:
                return httpx.Response(503)
            return httpx.Response(200, json={"message": {"content": "ok"}})
        with patch.multiple(config, LLM_PROVIDER="ollama", LLM_RETRIES=1), patch("llm.qwen.time.sleep"):
            self.assertEqual(self.client(handler).chat([]), "ok")
            self.assertEqual(len(calls), 2)
            calls.clear()
            def unauthorized(req):
                calls.append(req)
                return httpx.Response(401, text="sensitive server text")
            with self.assertRaises(LLMError) as raised:
                self.client(unauthorized).chat([])
            self.assertEqual(len(calls), 1)
            self.assertNotIn("sensitive", str(raised.exception))

    def test_auto_retrieval_selects_dense_and_falls_back(self):
        from rag.retrieval import Retriever
        from pathlib import Path
        with patch.dict(os.environ, {"RETRIEVER_MODE": "auto", "LLM_BASE_URL": "https://model.invalid"}):
            with patch.object(Retriever, "_make_embedder", side_effect=RuntimeError("offline")) as factory:
                r = Retriever(kb_path=Path(__file__).resolve().parents[1] / "knowledge-base", mode="auto")
                factory.assert_called_once_with("dense")
                self.assertEqual(r.mode, "bm25")
