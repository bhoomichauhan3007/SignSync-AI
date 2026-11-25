# Test Member 2 → Member 3 integration

from member3_adapter import transformer_prediction_to_sentence

# Simulated predictions from Member 2
predictions = ["I", "WANT", "WATER"]

# Convert predictions into a sentence
result = transformer_prediction_to_sentence("WATER")

print("Member 2 prediction:", "WATER")
print("Member 3 sentence:", result)