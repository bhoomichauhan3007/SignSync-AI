# Test Member 3: Speech to Text using Whisper

from gtts import gTTS
from member3_adapter import speech_to_text

# Create sample English audio
text = "Hello, I need water. Please help me."
gTTS(text=text, lang="en").save("test_audio.mp3")

# Convert audio into bytes
with open("test_audio.mp3", "rb") as audio:
    audio_bytes = audio.read()

# Run Speech-to-Text
print("Transcribing audio...")
result = speech_to_text(audio_bytes, language="en")

print("Original:", text)
print("Recognized:", result)