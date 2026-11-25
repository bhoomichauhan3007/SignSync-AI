# Test Speech → Text → ISL Sign sequence

from member3_adapter import (
    speech_to_text,
    text_to_sign_sequence
)
from pathlib import Path

# Load your recorded voice
audio_bytes = Path("my_voice.m4a").read_bytes()

# Step 1: Speech to Text
text = speech_to_text(audio_bytes, language="en")
print("Recognized Text:", text)

# Step 2: Text to ISL Sign Sequence
signs = text_to_sign_sequence(text)

print("\nGenerated ISL Signs:")
for sign in signs:
    print(sign)