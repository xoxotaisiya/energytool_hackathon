"""Проверки подготовки базы знаний и локального поиска."""

import unittest
from pathlib import Path

from rag.build_chunks import load_docs, make_chunks
from rag.retriever import BM25Index
from rag.retrieval import parse_dialog, build_query


ROOT = Path(__file__).resolve().parents[1]


class KnowledgeBaseTests(unittest.TestCase):
    """Проверяет реальные документы и поисковый индекс проекта."""

    @classmethod
    def setUpClass(cls):
        docs = load_docs(ROOT / "knowledge-base")
        cls.chunks = make_chunks(docs)[0]
        cls.index = BM25Index(cls.chunks)

    def test_knowledge_base_is_loaded(self):
        """Проверяет, что исходная база содержит документы и фрагменты."""
        self.assertGreaterEqual(len(self.chunks), 20)
        self.assertTrue(any("Energytool" in c["text"] or c["path"] for c in self.chunks))

    def test_bm25_finds_relevant_product_material(self):
        """Проверяет поиск материала по вопросу о платформе учёта."""
        hits = self.index.search("облачная платформа автоматический учет электроэнергии", 5)
        self.assertTrue(hits)
        texts = " ".join(self.chunks[i]["path"] + " " + self.chunks[i]["text"] for i, _ in hits)
        self.assertTrue(any(word in texts.lower() for word in ("учет", "учёт", "платформ")))

    def test_dialog_history_is_parsed(self):
        """Проверяет разбор истории диалога на реплики."""
        turns = parse_dialog(
            "Пользователь: Нужен учёт электроэнергии\n"
            "Ассистент: Расскажите о задаче\n"
            "Пользователь: Для нескольких объектов"
        )
        self.assertEqual(len(turns), 3)
        self.assertEqual(turns[-1][0], "user")
        self.assertIn("нескольких объектов", build_query(turns))


if __name__ == "__main__":
    unittest.main()
