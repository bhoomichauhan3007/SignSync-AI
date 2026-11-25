# Test Member 3: Text to Speech using gTTS

from member3_adapter import text_to_speech

# Input text
text = "Hello, I need water. Please help me."

# Generate speech audio
audio_bytes = text_to_speech(text, language="en")

# Save generated audio
with open("member3_tts_test.mp3", "wb") as f:
    f.write(audio_bytes)

print("Text to Speech successful!")
print("Audio file size:", len(audio_bytes), "bytes")
print("Saved as: member3_tts_test.mp3")