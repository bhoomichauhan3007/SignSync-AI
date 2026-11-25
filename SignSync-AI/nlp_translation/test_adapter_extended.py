
import unittest
from unittest.mock import patch

import member3_adapter


class TestAdapterExtended(unittest.TestCase):

    @patch("member3_adapter.signs_to_sentence")
    def test_transformer_prediction(self, mock_sentence):
        mock_sentence.return_value = "Water."

        result = member3_adapter.transformer_prediction_to_sentence(
            "WATER"
        )

        self.assertEqual(result, "Water.")
        mock_sentence.assert_called_once_with(["WATER"])

    def test_empty_transformer_prediction(self):
        result = member3_adapter.transformer_prediction_to_sentence("")
        self.assertEqual(result, "")

    @patch("member3_adapter.isl_nlp.speech_to_text")
    def test_speech_to_text(self, mock_stt):
        mock_stt.return_value = "Hello"

        result = member3_adapter.speech_to_text(
            b"fake audio data", "en"
        )

        self.assertEqual(result, "Hello")

    @patch("member3_adapter.isl_nlp.text_to_audio_file")
    def test_text_to_speech(self, mock_tts):

        def fake_tts(text, path, lang):
            with open(path, "wb") as f:
                f.write(b"fake audio")

        mock_tts.side_effect = fake_tts

        result = member3_adapter.text_to_speech(
            "Hello", "en"
        )

        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
