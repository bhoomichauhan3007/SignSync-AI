# Check vocabulary coverage for a sample sentence

import json
import re

with open("sign_vocab.json", encoding="utf-8") as f:
    vocab = json.load(f)

sentence = "Hello my name is Charu I need water please help me"

words = re.findall(r"[A-Za-z]+", sentence.upper())

print("WORD-BY-WORD VOCABULARY CHECK\n")

for word in words:
    if word in vocab:
        print(word, "-> AVAILABLE:", vocab[word])
    else:
        print(word, "-> MISSING: alphabet fallback may be used")

print("\nVocabulary entries:", len(vocab))