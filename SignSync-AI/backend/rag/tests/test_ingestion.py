import tempfile
import unittest
from pathlib import Path

from ._helpers import make_kb, make_pdf
from ..ingestion.chunking import chunk_document
from ..ingestion.cleaning import clean_text
from ..ingestion.loaders import Document, iter_documents, status_for


class CleaningTests(unittest.TestCase):
    def test_removes_page_markers_and_joins_hyphenated_words(self):
        raw = "Learning ISL is use-\nful every day.\n\nPage 3 of 10\n\n\n\nNext   paragraph\there."
        cleaned = clean_text(raw)
        self.assertIn("useful every day.", cleaned)
        self.assertNotIn("Page 3 of 10", cleaned)
        self.assertIn("Next paragraph here.", cleaned)
        self.assertNotIn("\n\n\n", cleaned)

    def test_keeps_numbers_that_are_not_page_markers(self):
        self.assertIn("2024", clean_text("The year 2024"))
        self.assertIn("\n1\n", clean_text("list\n1\nitem"))

    def test_normalises_unicode_and_soft_hyphens(self):
        self.assertEqual(clean_text("ﬁne\u00ad print"), "fine print")


class ChunkingTests(unittest.TestCase):
    def _doc(self, text, page=None):
        return Document(text=text, source="sample/x.md", title="X", page=page, status="sample")

    def test_chunks_respect_size_and_keep_metadata(self):
        sentence = "Indian Sign Language uses handshape, movement and location. "
        chunks = chunk_document(self._doc(sentence * 60, page=2), max_chars=400, overlap_chars=50)
        self.assertGreater(len(chunks), 3)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), 400 + 50)  # max plus allowed overlap tail
            self.assertEqual((chunk.source, chunk.title, chunk.page, chunk.status), ("sample/x.md", "X", 2, "sample"))

    def test_chunk_ids_are_stable_across_runs(self):
        doc = self._doc("Alpha beta gamma. " * 80)
        first = [c.id for c in chunk_document(doc, 300, 40)]
        second = [c.id for c in chunk_document(doc, 300, 40)]
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), len(first))

    def test_overlap_carries_text_between_neighbouring_chunks(self):
        text = "\n\n".join(f"Paragraph number {i} talks about sign {i}." for i in range(30))
        chunks = chunk_document(self._doc(text), max_chars=300, overlap_chars=60)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(any(
            chunks[i].text.split()[-1] in chunks[i + 1].text for i in range(len(chunks) - 1)
        ))

    def test_empty_document_gives_no_chunks(self):
        self.assertEqual(chunk_document(self._doc("   \n\n  ")), [])

    def test_rejects_tiny_chunk_size(self):
        with self.assertRaises(ValueError):
            chunk_document(self._doc("text"), max_chars=50)

    def test_very_long_sentence_is_split(self):
        chunks = chunk_document(self._doc("word " * 800), max_chars=300, overlap_chars=0)
        self.assertTrue(all(len(c.text) <= 300 for c in chunks))


class LoaderTests(unittest.TestCase):
    def test_status_comes_from_folder(self):
        self.assertEqual(status_for(Path("verified/a.pdf")), "verified")
        self.assertEqual(status_for(Path("sample/b.md")), "sample")
        self.assertEqual(status_for(Path("c.txt")), "unverified")
        self.assertEqual(status_for(Path("other/d.txt")), "unverified")

    def test_iter_documents_skips_docs_readme_and_hidden_files(self):
        kb = make_kb({
            "verified/guide.md": "# ISL Guide\n\nHello there.",
            "SOURCES.md": "# sources",
            "README.md": "# readme",
            ".hidden/secret.txt": "nope",
            "notes.csv": "a,b",
            "unverified/tips.txt": "Practise daily.",
        })
        docs = list(iter_documents(kb))
        sources = sorted(d.source for d in docs)
        self.assertEqual(sources, ["unverified/tips.txt", "verified/guide.md"])
        guide = next(d for d in docs if d.source == "verified/guide.md")
        self.assertEqual(guide.title, "ISL Guide")
        self.assertEqual(guide.status, "verified")
        self.assertIsNone(guide.page)

    def test_pdf_loader_gives_one_document_per_page_with_page_numbers(self):
        kb = make_kb({"verified/course.pdf": make_pdf(["Greeting hello sign", "Emergency help phrase"])})
        docs = list(iter_documents(kb))
        self.assertEqual([d.page for d in docs], [1, 2])
        self.assertIn("Greeting", docs[0].text)
        self.assertIn("Emergency", docs[1].text)
        self.assertEqual(docs[0].title, "course")
        self.assertEqual(docs[0].status, "verified")

    def test_missing_folder_raises_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            list(iter_documents(Path(tempfile.gettempdir()) / "does_not_exist_rag_kb"))


if __name__ == "__main__":
    unittest.main()
