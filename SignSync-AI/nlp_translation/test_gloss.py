
import json
import isl_nlp
from gemini_nlp import text_to_gloss

with open("sign_vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

sentences = [
    "Hello, I need water, please help me.",
    "Good morning.",
    "I want food.",
    "My name is Charu."
]

for sentence in sentences:
    print("\nInput:", sentence)

    gloss = text_to_gloss(sentence, vocab)
    print("Generated Gloss:", gloss)

    clips = isl_nlp.gloss_to_clips(gloss, vocab)
    print("Mapped Clips:", clips)
