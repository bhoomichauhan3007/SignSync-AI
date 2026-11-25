
# Test Member 3: Gemini NLP functions

from gemini_nlp import (
    words_to_sentence,
    extract_intent,
    text_to_gloss
)

# Small test vocabulary
vocab = {
    "HELLO": "hello.mp4",
    "WATER": "water.mp4",
    "THANKS": "thanks.mp4",
    "HELP": "help.mp4",
    "DOCTOR": "doctor.mp4",
    "HOSPITAL": "hospital.mp4"
}

# TEST 1: ISL words to sentence
print("\nTEST 1: Words to Sentence")
print(words_to_sentence(["I", "WANT", "WATER"]))

# TEST 2: Emergency intent
print("\nTEST 2: Emergency Intent")
print(extract_intent("Please help me, I need a doctor."))

# TEST 3: Text to ISL gloss
print("\nTEST 3: Text to Gloss")
print(text_to_gloss("Hello, I want water.", vocab))

# TEST 4: Empty input
print("\nTEST 4: Empty Input")
print("Result:", repr(words_to_sentence([])))

# TEST 5: Question intent
print("\nTEST 5: Question Intent")
print(extract_intent("Where is the hospital?"))

# TEST 6: Request intent
print("\nTEST 6: Request Intent")
print(extract_intent("Please give me water."))

# TEST 7: Normal statement
print("\nTEST 7: Normal Statement")
print(words_to_sentence(["I", "FEEL", "HAPPY", "TODAY"]))

# TEST 8: Longer sentence
print("\nTEST 8: Longer Sentence")
print(words_to_sentence([
    "I", "WANT", "TO", "GO", "TO", "HOSPITAL"
]))

# TEST 9: Emergency sentence
print("\nTEST 9: Emergency Sentence")
print(extract_intent("I am injured and need medical help immediately."))

# TEST 10: Another question
print("\nTEST 10: Question Classification")
print(extract_intent("Can you tell me where the railway station is?"))

# TEST 11: Another request
print("\nTEST 11: Request Classification")
print(extract_intent("Could you please bring me some water?"))

print("\nALL TEST CASES EXECUTED")
