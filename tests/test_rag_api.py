"""Integration tests for Flask RAG endpoints and backwards compatibility."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import app


class FlaskRAGApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.app.test_client()
        from src.rag.config import load_rag_config
        from src.rag.rag_pipeline import RAGPipeline
        config=load_rag_config();config.llm.provider='offline_grounded'
        cls.pipeline=RAGPipeline(config)
        guard=patch('src.rag.get_rag_pipeline',return_value=cls.pipeline)
        guard.start();cls.addClassCleanup(guard.stop)

    def test_health_endpoint_unchanged(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["model_loaded"])
        self.assertTrue(data["ood_detector_loaded"])
        self.assertEqual(data["num_classes"], 120)

    def test_rag_status_endpoint(self):
        res = self.client.get("/api/rag/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["knowledge_version"].startswith("2026-10-17-v2-"))
        self.assertGreater(data["chunks_count"], 0)
        self.assertGreater(data["files_count"], 0)
        self.assertIn(data["llm_status"], ["live", "missing_config"])

    def test_rag_query_requires_json(self):
        res = self.client.post("/api/rag/query", data="not json", content_type="text/plain")
        self.assertEqual(res.status_code, 400)
        self.assertIn("JSON", res.get_json()["error"])

    def test_rag_query_empty_query_rejected(self):
        res = self.client.post("/api/rag/query", json={"query": "   "})
        self.assertEqual(res.status_code, 400)
        self.assertIn("non-empty", res.get_json()["error"])

    def test_rag_query_invalid_types_rejected(self):
        res1 = self.client.post("/api/rag/query", json={"query": "test", "history": "not a list"})
        self.assertEqual(res1.status_code, 400)
        res2 = self.client.post("/api/rag/query", json={"query": "test", "last_prediction": "not an object"})
        self.assertEqual(res2.status_code, 400)

    def test_rag_query_successful_response(self):
        res = self.client.post("/api/rag/query", json={"query": "Mô hình hỗ trợ bao nhiêu giống?"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("answer", data)
        self.assertIn("120", data["answer"])
        self.assertIn("sources", data)
        self.assertTrue(isinstance(data["sources"], list))
        self.assertTrue(data["knowledge_version"].startswith("2026-10-17-v2-"))
        self.assertFalse(data["missing_evidence"])

    def test_source_endpoint_does_not_read_arbitrary_files(self):
        self.assertEqual(self.client.get('/api/rag/source?chunk_id=../../.env').status_code,404)

    def test_source_endpoint_returns_exact_indexed_excerpt(self):
        from src.rag import get_rag_pipeline
        chunk=get_rag_pipeline().indexer.source_chunks('README.md',1)[0]
        response=self.client.get('/api/rag/source',query_string={'chunk_id':chunk.chunk_id})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.get_json()['content'],chunk.content)

    def test_rag_query_rejects_array_and_invalid_history_turn(self):
        self.assertEqual(self.client.post('/api/rag/query',json=['query']).status_code,400)
        self.assertEqual(self.client.post('/api/rag/query',json={'query':'test','history':[3]}).status_code,400)

    def test_rag_query_with_prediction_context(self):
        res = self.client.post("/api/rag/query", json={
            "query": "Tại sao ảnh này lại có kết quả là UNCERTAIN?",
            "last_prediction": {
                "status": "UNCERTAIN",
                "breed": "collie",
                "confidence": 0.45,
                "ood_score": 0.38,
            }
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("UNCERTAIN", data["answer"])
        self.assertIn("collie", data["answer"])
        self.assertFalse(data["missing_evidence"])


if __name__ == "__main__":
    unittest.main()
