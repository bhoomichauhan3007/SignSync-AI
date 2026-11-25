// Text & Voice: type or speak a sentence, see it as signs, and hear it read aloud.
// Sign pictures come from the gallery: letters use the reference photos, words use the picture cards.
import { useState } from "react";
import { ArrowRight, Mic, MicOff, Send, Volume2 } from "lucide-react";
import { ApiError, speechToSign, textToSign } from "./api";
import { listenOnce, recordClip, speakText, voiceInputSupported } from "./speech";
import { signPictureUrl, PICTURE_EXTENSIONS, isVideo } from "./signs";

const EXAMPLES = ["Hello, how are you?", "I need water, please", "Thank you, good morning", "I love you"];

function SignCard({ label }) {
  const isLetter = label.length === 1;
  const kind = isLetter ? "alphabet" : "words";
  const [attempt, setAttempt] = useState(0);
  const [failed, setFailed] = useState(false);
  const shown = label.toLowerCase();
  return (
    <figure className="sign-card">
      <div className="sign-card-picture">
        {failed ? (
          <span className="sign-card-text">{label}</span>
        ) : (
          isVideo(PICTURE_EXTENSIONS[attempt]) ? (
            <video
              key={signPictureUrl(kind, shown, attempt)}
              src={signPictureUrl(kind, shown, attempt)}
              aria-label={`Sign video for ${label}`}
              autoPlay
              loop
              muted
              playsInline
              onError={() => (attempt + 1 < PICTURE_EXTENSIONS.length ? setAttempt(attempt + 1) : setFailed(true))}
            />
          ) : (
            <img
              src={signPictureUrl(kind, shown, attempt)}
              alt={`Sign for ${label}`}
              loading="lazy"
              onError={() => (attempt + 1 < PICTURE_EXTENSIONS.length ? setAttempt(attempt + 1) : setFailed(true))}
            />
          )
        )}
      </div>
      <figcaption>{isLetter ? label : label.charAt(0) + label.slice(1).toLowerCase()}</figcaption>
    </figure>
  );
}

export default function TextVoice({ token }) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [listening, setListening] = useState(false);
  const [recognizer, setRecognizer] = useState(null);
  const [speaking, setSpeaking] = useState(false);
  const [clip, setClip] = useState(null);      // active recording, see recordClip()
  const [whisperBusy, setWhisperBusy] = useState(false);

  // Records a short clip and sends it to the server, where Whisper (Member 3) turns it into text.
  const whisperToggle = async () => {
    setError("");
    if (clip) { clip.stop(); return; }
    try {
      const next = recordClip(8);
      setClip(next);
      const blob = await next.done;
      setClip(null);
      setWhisperBusy(true);
      const data = await speechToSign(blob, token);
      setText(data.text);
      setResult(data);
    } catch (err) {
      setClip(null);
      setError(err instanceof ApiError ? err.message : (err.message || "Voice could not be understood. Please try again."));
    } finally {
      setWhisperBusy(false);
    }
  };

  const translate = async event => {
    event?.preventDefault();
    const clean = text.trim();
    if (!clean) return;
    setBusy(true);
    setError("");
    try {
      setResult(await textToSign(clean, token));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "The text could not be translated. Please try again.");
      setResult(null);
    } finally {
      setBusy(false);
    }
  };

  const toggleListening = () => {
    if (listening) {
      recognizer?.stop();
      setListening(false);
      return;
    }
    setError("");
    const rec = listenOnce({
      onResult: transcript => {
        setText(transcript);
        setListening(false);
        setTimeout(() => translate(), 50);
      },
      onError: message => { setError(message); setListening(false); },
      onEnd: () => setListening(false),
    });
    if (rec) {
      setRecognizer(rec);
      setListening(true);
    }
  };

  const read = async () => {
    setSpeaking(true);
    await speakText(text.trim() || result?.text || "", token);
    setSpeaking(false);
  };

  return (
    <section className="page-content text-voice-page">
      <div className="live-hero">
        <div>
          <span className="eyebrow">TEXT AND VOICE</span>
          <h1>Text & Voice to Sign</h1>
          <p>Type a sentence or speak it. See each sign, then hear the sentence read aloud.</p>
        </div>
      </div>

      <div className="tv-grid">
        <form className="panel tv-input" onSubmit={translate}>
          <label className="tv-label" htmlFor="tv-text">Your sentence</label>
          <textarea
            id="tv-text"
            rows={4}
            maxLength={1000}
            value={text}
            onChange={e => setText(e.target.value)}
            placeholder="Type here, for example: I need water, please"
          />
          <div className="tv-examples">
            {EXAMPLES.map(example => (
              <button type="button" key={example} className="chip-button" onClick={() => setText(example)}>{example}</button>
            ))}
          </div>
          <div className="tv-actions">
            <button type="button" className={`mic-button ${listening ? "is-listening" : ""}`} onClick={toggleListening}
              disabled={!voiceInputSupported()} title={voiceInputSupported() ? "Speak your sentence" : "Voice typing works in Chrome or Edge"}>
              {listening ? <MicOff size={17} /> : <Mic size={17} />} {listening ? "Listening..." : "Speak"}
            </button>
            <button type="button" className={`mic-button ${clip ? "is-listening" : ""}`} onClick={whisperToggle} disabled={whisperBusy}
              title="Record up to 8 seconds. The server transcribes it with Whisper.">
              {clip ? <MicOff size={17} /> : <Mic size={17} />} {clip ? "Stop recording" : whisperBusy ? "Transcribing..." : "Record (Whisper)"}
            </button>
            <button type="submit" className="primary-button" disabled={busy || !text.trim()}>
              <Send size={16} /> {busy ? "Translating..." : "Translate to signs"}
            </button>
          </div>
          {!voiceInputSupported() && <p className="hint-message">Voice typing needs Chrome or Edge. You can still type.</p>}
          {error && <p className="error-message" role="alert">{error}</p>}
        </form>

        <div className="panel tv-output">
          <div className="tv-output-head">
            <strong>Signs</strong>
            <button type="button" className="secondary-button" onClick={read} disabled={speaking || (!text.trim() && !result)}>
              <Volume2 size={15} /> {speaking ? "Reading..." : "Read aloud"}
            </button>
          </div>
          {!result && !busy && (
            <div className="tv-empty">
              <div className="tv-empty-icon"><Send size={26} /></div>
              <p>Your signs will appear here, one picture for each word or letter.</p>
            </div>
          )}
          {busy && <div className="tv-empty"><p>Translating...</p></div>}
          {result && (
            <>
              <div className="sign-flow">
                {result.sequence.map((item, index) => (
                  <div className="sign-flow-item" key={`${item.label}-${index}`}>
                    <SignCard label={item.label} />
                    {index < result.sequence.length - 1 && <ArrowRight size={16} className="flow-arrow" />}
                  </div>
                ))}
              </div>
              <p className="sentence-source">
                {result.source === "real" ? "Translated with the sentence engine (Gemini)." : "Translated with the built-in rules."}
              </p>
            </>
          )}
        </div>
      </div>
    </section>
  );
}
