import os
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
from rag.fusion import rrf
from rag.retrieval import Retriever
from rag.reranker import ApiReranker
from rag.embedder import ApiEmbedder
from rag.ollama_reranker import OllamaReranker

ROOT = Path(__file__).resolve().parents[1]


class RetrievalModesTests(unittest.TestCase):
    def test_local_reranker_requests_bounded_json_schema(self):
        import json
        rr = OllamaReranker()
        self.addCleanup(rr.close)
        def handler(request):
            payload = json.loads(request.content)
            schema = payload["format"]
            self.assertEqual(schema["required"], ["indices"])
            self.assertEqual(schema["properties"]["indices"]["items"]["enum"], [0, 1])
            self.assertEqual(schema["properties"]["indices"]["maxItems"], 1)
            return httpx.Response(200, json={"done_reason": "stop", "message": {"content": '{"indices":[1]}'}})
        rr.client.close()
        rr.client = httpx.Client(transport=httpx.MockTransport(handler))
        self.assertEqual(rr.rerank("q", [{"path": "a", "text": "a"}, {"path": "b", "text": "b"}], 1), [1])

    def test_rrf_weights_and_duplicates(self):
        hits = rrf([[(1, 100), (1, 90), (2, 80)], [(2, .9), (3, .8)]], [.4, .6])
        self.assertEqual(len(hits), 3)
        self.assertEqual(hits[0][0], 2)
        self.assertAlmostEqual(dict(hits)[1], .4 / 61)

    def test_hybrid_uses_both_rankings_and_preserves_metadata(self):
        # Реальная база, управляемые векторные результаты, без моделей/сети.
        r = Retriever(kb_path=ROOT / "knowledge-base", mode="bm25")
        r.mode = "hybrid"
        r.embedder = object()
        with patch.object(r, "_dense", return_value=[(0, .8), (1, .7)]), \
             patch.object(r.bm25, "search", return_value=[(1, 50), (2, 30)]):
            result = r.retrieve("test", top_k=3)
        self.assertEqual(result["mode"], "hybrid")
        self.assertEqual(result["score_kind"], "rrf")
        self.assertEqual(result["confidence"], .8)
        self.assertEqual(result["chunks"][0]["id"], r.chunks[1]["id"])
        self.assertEqual(len({c["id"] for c in result["chunks"]}), 3)
        self.assertTrue(all("file" in c for c in result["chunks"]))

    def test_query_embedding_failure_falls_back(self):
        r = Retriever(kb_path=ROOT / "knowledge-base", mode="bm25")
        r.mode, r.embedder = "hybrid", object()
        with patch.object(r, "_dense", side_effect=RuntimeError("offline")):
            result = r.retrieve("солнечная станция")
        self.assertEqual(result["mode"], "bm25")
        self.assertIsNone(result["confidence"])
        self.assertTrue(result["chunks"])

    def test_incomplete_embeddings_not_accepted(self):
        with patch.dict(os.environ, {"LLM_BASE_URL": "https://model.invalid"}):
            embedder = ApiEmbedder()
        self.addCleanup(embedder.client.close)
        response = httpx.Response(200, request=httpx.Request("POST", "https://model.invalid/embeddings"),
                                  json={"data": [{"index": 1, "embedding": [1., 0.]}]})
        with patch.object(embedder.client, "post", return_value=response), patch("rag.embedder.time.sleep"):
            with self.assertRaises(ValueError):
                embedder.encode(["a", "b"])

    def test_reranker_validates_indices(self):
        with patch.dict(os.environ, {"RERANKER_BASE_URL": "https://model.invalid"}):
            reranker = ApiReranker()
        self.addCleanup(reranker.close)
        response = httpx.Response(200, request=httpx.Request("POST", "https://model.invalid/rerank"),
                                  json={"results": [{"index": 1}, {"index": 0}]})
        with patch.object(reranker.client, "post", return_value=response):
            self.assertEqual(reranker.rerank("q", [{"path": "a", "text": "a"}, {"path": "b", "text": "b"}], 1), [1])
        response = httpx.Response(200, request=httpx.Request("POST", "https://model.invalid/rerank"),
                                  json={"results": [{"index": 2}]})
        with patch.object(reranker.client, "post", return_value=response):
            with self.assertRaises(ValueError):
                reranker.rerank("q", [{"path": "a", "text": "a"}], 1)

    def test_reranker_failure_keeps_original_order(self):
        r = Retriever(kb_path=ROOT / "knowledge-base", mode="bm25")
        before = r.retrieve("солнечная станция")
        class Failed:
            def rerank(self, *args):
                raise RuntimeError("offline")
        r.reranker = Failed()
        after = r.retrieve("солнечная станция")
        self.assertEqual([c["id"] for c in before["chunks"]], [c["id"] for c in after["chunks"]])
        self.assertFalse(after["reranked"])
        self.assertEqual(after["rerank_error"], "RuntimeError")

    def test_partial_reranking_does_not_drop_remaining_candidates(self):
        r = Retriever(kb_path=ROOT / "knowledge-base", mode="bm25")
        before = r.retrieve("солнечная станция", top_k=5)
        class Partial:
            def rerank(self, *args):
                return [1]
        r.reranker = Partial()
        after = r.retrieve("солнечная станция", top_k=5)
        self.assertEqual(len(after["chunks"]), 5)
        self.assertEqual(after["chunks"][0]["id"], before["chunks"][1]["id"])
        self.assertEqual(len({c["id"] for c in after["chunks"]}), 5)
