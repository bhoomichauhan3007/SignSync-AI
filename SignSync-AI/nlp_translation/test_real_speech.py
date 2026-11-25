# Test real human voice using Whisper

from pathlib import Path
from member3_adapter import speech_to_text

# Load recorded M4A audio
audio_bytes = Path("my_voice.m4a").read_bytes()

# Convert speech to text
print("Transcribing your voice...")

result = speech_to_text(audio_bytes, language="en")

print("\nRecognized Speech:")
print(result)