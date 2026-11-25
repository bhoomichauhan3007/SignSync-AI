
import json
import unittest
from unittest.mock import patch

import gemini_nlp
import isl_nlp


class TestMember3Regression(unittest.TestCase):

    def setUp(self):
        with open("sign_vocab.json", "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

    def test_empty_text(self):
        result = gemini_nlp.text_to_gloss("", self.vocab)
        self.assertEqual(result, [])

    @patch("gemini_nlp.generate_text", return_value="I FOOD")
    def test_missing_word_preservation(self, mock_gemini):
        result = gemini_nlp.text_to_gloss(
            "I want food", self.vocab
        )
        self.assertIn("WANT", result)

    @patch("gemini_nlp.generate_text", return_value="C H A R U NAME")
    def test_name_fingerspelling_not_duplicated(self, mock_gemini):
        result = gemini_nlp.text_to_gloss(
            "My name is Charu", self.vocab
        )

        for letter in ["C", "H", "A", "R", "U"]:
            self.assertEqual(result.count(letter), 1)

    def test_word_and_alphabet_mapping(self):
        result = isl_nlp.gloss_to_clips(
            ["WATER", "A"], self.vocab
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["type"], "word")
        self.assertEqual(result[1]["type"], "letter")

    def test_unknown_word_fingerspelling(self):
        result = isl_nlp.gloss_to_clips(
            ["CAT"], self.vocab
        )

        self.assertEqual(
            [item["gloss"] for item in result],
            ["C", "A", "T"]
        )

    def test_vocabulary_loaded(self):
        self.assertGreaterEqual(len(self.vocab), 60)


if __name__ == "__main__":
    unittest.main(verbosity=2)
