// Speech for SignSync: reading text aloud and typing by voice.
// 1) Server voice (Member 3's gTTS) if it is installed on the backend.
// 2) The browser's built-in voice, which needs no installation (works in Chrome, Edge, Safari, Firefox).
// Voice input uses the browser's speech recognition (Chrome and Edge). Chrome sends the audio to Google.

import { textToSpeechBlob } from "./api";

export function speakWithBrowser(text, lang = "en-IN") {
  if (!text || !("speechSynthesis" in window)) return false;
  const synth = window.speechSynthesis;
  synth.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  const voices = synth.getVoices();
  const voice = voices.find(v => v.lang === lang) || voices.find(v => v.lang.startsWith("en"));
  if (voice) utterance.voice = voice;
  utterance.lang = lang;
  utterance.rate = 0.95;
  synth.speak(utterance);
  return true;
}

// Reads text aloud. Resolves with "server", "browser" or "none".
export async function speakText(text, token) {
  if (!text) return "none";
  try {
    const blob = await textToSpeechBlob(text, token);
    if (blob) {
      const audio = new Audio(URL.createObjectURL(blob));
      await audio.play();
      return "server";
    }
  } catch {
    /* fall through to the browser voice */
  }
  return speakWithBrowser(text) ? "browser" : "none";
}

export function voiceInputSupported() {
  return Boolean(window.SpeechRecognition || window.webkitSpeechRecognition);
}

// Listens once and returns the recogniser (call .stop() to cancel).
export function listenOnce({ onResult, onError, onEnd, lang = "en-IN" }) {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) {
    onError("Voice typing works in Chrome or Edge. You can type the text instead.");
    return null;
  }
  const recognition = new Recognition();
  recognition.lang = lang;
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;
  recognition.onresult = event => onResult(event.results[0][0].transcript);
  recognition.onerror = event => onError(event.error === "not-allowed"
    ? "Microphone access was blocked. Allow the microphone in your browser and try again."
    : "Voice typing stopped. Please try again.");
  recognition.onend = () => onEnd && onEnd();
  recognition.start();
  return recognition;
}

// Records a short voice clip with the microphone. Resolves with a Blob (or rejects with a readable message).
// Call the returned object's stop() to finish; the promise then resolves with the audio.
export function recordClip(maxSeconds = 8) {
  let recorder = null;
  let chunks = [];
  let timer = null;
  const done = new Promise((resolve, reject) => {
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(stream => {
        const type = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"].find(t => window.MediaRecorder?.isTypeSupported?.(t)) || "";
        recorder = new MediaRecorder(stream, type ? { mimeType: type } : undefined);
        recorder.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
        recorder.onstop = () => {
          stream.getTracks().forEach(track => track.stop());
          clearTimeout(timer);
          resolve(new Blob(chunks, { type: recorder.mimeType || "audio/webm" }));
        };
        chunks = [];
        recorder.start();
        timer = setTimeout(() => recorder.state !== "inactive" && recorder.stop(), maxSeconds * 1000);
      })
      .catch(err => reject(new Error(err?.name === "NotAllowedError"
        ? "Microphone access was blocked. Allow the microphone in your browser and try again."
        : "The microphone could not be started.")));
  });
  return {
    done,
    stop: () => { if (recorder && recorder.state !== "inactive") recorder.stop(); },
  };
}
