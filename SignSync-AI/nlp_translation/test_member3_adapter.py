
# Test Member 3 backend adapter

from member3_adapter import (
    signs_to_sentence,
    text_to_sign_sequence
)

# Test 1: ISL words to sentence
print("\nTEST 1: Signs to Sentence")
print(signs_to_sentence(["I", "WANT", "WATER"]))

# Test 2: Text to sign sequence
print("\nTEST 2: Text to Sign Sequence")
result = text_to_sign_sequence("Hello")

print(result)

# Verify backend response structure
if result:
    assert "label" in result[0]
    assert "animation_id" in result[0]
    print("\nBackend response format: VALID")
else:
    print("\nNo sign clips returned. Check sign_vocab.json.")
