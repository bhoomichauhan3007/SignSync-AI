// Live Translator: the camera reads hands (letters) or the whole body (words) in the browser.
// Letters are recognised in the browser with Member 2's alphabet model, so there is no network delay.
// Words are sent to the server's word model. The sentence is formed by Member 3's sentence module.
import { useEffect, useRef, useState } from "react";
import { Camera, Check, Hand, Languages, Sparkles, Square, Target, Trash2, Volume2 } from "lucide-react";
import { ApiError, predictSign, signsToSentence } from "./api";
import { holisticFrameVector, hasAnyHand } from "./holisticFeatures";
import { LetterVoter } from "./liveLogic";
import { loadHolistic, createHolistic, loadHands, createHands } from "./mediapipeLoader";
import { loadAlphabetModel, classifyHand, handToVector } from "./alphabetModel";
import { speakText } from "./speech";
import { WORDS } from "./signs";

// The model answers with lowercase names such as "iloveyou"; show the friendly text ("I love you").
const wordText = label => WORDS.find(w => w.gloss === label.toUpperCase())?.text ?? titleCase(label);

const FRAMES = 30;               // frames per word (Member 2's word model length)
const HAND_LINKS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];
const POSE_LINKS = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24]];

function drawSkeleton(ctx, points, links, w, h, color, width) {
  if (!points) return;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  for (const [a, b] of links) {
    const p = points[a];
    const q = points[b];
    if (!p || !q || (p.visibility ?? 1) < 0.3 || (q.visibility ?? 1) < 0.3) continue;
    ctx.beginPath();
    ctx.moveTo(p.x * w, p.y * h);
    ctx.lineTo(q.x * w, q.y * h);
    ctx.stroke();
  }
  for (const p of points) {
    if ((p.visibility ?? 1) < 0.3) continue;
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, width + 1, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawOverlay(canvas, video, { hands, holistic }) {
  if (!canvas || !video) return;
  const w = video.videoWidth || 640;
  const h = video.videoHeight || 480;
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  if (holistic) {
    drawSkeleton(ctx, holistic.poseLandmarks, POSE_LINKS, w, h, "rgba(255,255,255,0.7)", 2);
    drawSkeleton(ctx, holistic.leftHandLandmarks, HAND_LINKS, w, h, "#38bdf8", 3);
    drawSkeleton(ctx, holistic.rightHandLandmarks, HAND_LINKS, w, h, "#4ade80", 3);
  }
  if (hands?.multiHandLandmarks) {
    for (const hand of hands.multiHandLandmarks) drawSkeleton(ctx, hand, HAND_LINKS, w, h, "#4ade80", 3);
  }
}

function cameraErrorText(err) {
  if (err?.name === "NotAllowedError") return "Camera access was blocked. Allow the camera in your browser's address bar, then try again.";
  if (err?.name === "NotFoundError") return "No camera was found on this device.";
  if (err?.name === "NotReadableError") return "The camera is being used by another app. Close it and try again.";
  return err?.message || "The camera could not be started.";
}

const titleCase = text => text.charAt(0).toUpperCase() + text.slice(1).toLowerCase();
const percent = value => `${Math.round((value || 0) * 100)}%`;

export default function LiveTranslator({ token }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const runningRef = useRef(false);
  const busyRef = useRef(false);
  const modeRef = useRef("letters");
  const tokenRef = useRef(token);
  const handsRef = useRef(null);
  const holisticRef = useRef(null);
  const handsInit = useRef(null);
  const holisticInit = useRef(null);
  const modelRef = useRef(null);
  const voterRef = useRef(new LetterVoter({ windowSize: 7, needed: 5, minConfidence: 0.5, handGoneFrames: 4 }));
  const recordingRef = useRef(false);
  const framesRef = useRef([]);
  const lastFrameTime = useRef(0);
  const fpsCount = useRef(0);
  const fpsWindowStart = useRef(0);
  const handlerRefs = useRef({});

  const [mode, setMode] = useState("letters");
  const [phase, setPhase] = useState("idle");          // idle | starting | running | error
  const [error, setError] = useState("");
  const [engineNote, setEngineNote] = useState("");
  const [fps, setFps] = useState(0);
  const [handVisible, setHandVisible] = useState(false);
  const [letterLive, setLetterLive] = useState(null);  // { label, confidence, top }
  const [progress, setProgress] = useState({ label: null, count: 0 });
  const [spelling, setSpelling] = useState("");
  const [words, setWords] = useState([]);
  const [recording, setRecording] = useState(false);
  const [recordCount, setRecordCount] = useState(0);
  const [wordResult, setWordResult] = useState(null);
  const [wordBusy, setWordBusy] = useState(false);
  const [sentence, setSentence] = useState({ text: "", source: "" });
  const [sentenceBusy, setSentenceBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [flash, setFlash] = useState("");

  useEffect(() => { tokenRef.current = token; }, [token]);

  // ----- letters: runs entirely in the browser
  const handleHands = results => {
    drawOverlay(canvasRef.current, videoRef.current, { hands: results });
    const landmarks = results.multiHandLandmarks?.[0];
    setHandVisible(Boolean(landmarks));
    if (!landmarks) {
      voterRef.current.push(null, 0);
      setLetterLive(null);
      setProgress({ label: null, count: 0 });
      return;
    }
    if (!modelRef.current) return;
    const prediction = classifyHand(modelRef.current, handToVector(landmarks));
    setLetterLive(prediction);
    const accepted = voterRef.current.push(prediction.label, prediction.confidence);
    setProgress(voterRef.current.status());
    if (accepted) {
      setSpelling(text => text + accepted);
      setFlash(accepted);
      setTimeout(() => setFlash(""), 700);
    }
  };

  // ----- words: MediaPipe Holistic frames, recognised on the server
  const classifyWord = async sequence => {
    setWordBusy(true);
    setWordResult(null);
    try {
      setWordResult(await predictSign(sequence, "dynamic", tokenRef.current));
    } catch (err) {
      setWordResult({ failed: true, message: err instanceof ApiError ? err.message : "Recognition failed. Please try again." });
    } finally {
      setWordBusy(false);
    }
  };

  const handleHolistic = results => {
    drawOverlay(canvasRef.current, videoRef.current, { holistic: results });
    setHandVisible(hasAnyHand(results));
    if (!recordingRef.current) return;
    framesRef.current.push(holisticFrameVector(results));
    setRecordCount(framesRef.current.length);
    if (framesRef.current.length >= FRAMES) {
      const sequence = framesRef.current;
      framesRef.current = [];
      recordingRef.current = false;
      setRecording(false);
      classifyWord(sequence);
    }
  };

  handlerRefs.current = { hands: handleHands, holistic: handleHolistic };

  const countFrame = () => {
    const now = performance.now();
    fpsCount.current += 1;
    if (now - fpsWindowStart.current >= 1000) {
      setFps(fpsCount.current);
      fpsCount.current = 0;
      fpsWindowStart.current = now;
    }
  };

  const ensureHands = () => {
    if (handsRef.current) return Promise.resolve(handsRef.current);
    if (!handsInit.current) {
      handsInit.current = loadHands().then(Hands => {
        const hands = createHands(Hands);
        hands.onResults(results => { countFrame(); handlerRefs.current.hands?.(results); });
        handsRef.current = hands;
        return hands;
      });
    }
    return handsInit.current;
  };

  const ensureHolistic = () => {
    if (holisticRef.current) return Promise.resolve(holisticRef.current);
    if (!holisticInit.current) {
      holisticInit.current = loadHolistic().then(Holistic => {
        const holistic = createHolistic(Holistic);
        holistic.onResults(results => { countFrame(); handlerRefs.current.holistic?.(results); });
        holisticRef.current = holistic;
        return holistic;
      });
    }
    return holisticInit.current;
  };

  const loop = () => {
    const tick = async () => {
      if (!runningRef.current) return;
      const video = videoRef.current;
      if (video && video.readyState >= 2 && !busyRef.current) {
        const wantWords = modeRef.current === "words";
        const engine = wantWords ? holisticRef.current : handsRef.current;
        if (engine) {
          const now = performance.now();
          if (now - lastFrameTime.current > 16) {       // cap at about 60 frames per second
            lastFrameTime.current = now;
            busyRef.current = true;
            try {
              await engine.send({ image: video });
            } catch {
              /* skip this frame */
            } finally {
              busyRef.current = false;
            }
          }
        }
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  };

  const stopCamera = () => {
    runningRef.current = false;
    recordingRef.current = false;
    framesRef.current = [];
    setRecording(false);
    streamRef.current?.getTracks().forEach(track => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    voterRef.current.reset();
    setPhase("idle");
    setHandVisible(false);
    setLetterLive(null);
    setProgress({ label: null, count: 0 });
    setFps(0);
  };

  const startCamera = async () => {
    setError("");
    setNotice("");
    setPhase("starting");
    try {
      setEngineNote("Loading the letter model...");
      if (!modelRef.current) modelRef.current = await loadAlphabetModel();
      setEngineNote("Starting hand recognition...");
      await ensureHands();
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setEngineNote("");
      runningRef.current = true;
      fpsWindowStart.current = performance.now();
      setPhase("running");
      loop();
      if (modeRef.current === "words") ensureHolistic().then(() => setEngineNote(""));
    } catch (err) {
      setError(cameraErrorText(err) || (err instanceof Error ? err.message : "Could not start."));
      stopCamera();
      setPhase("error");
    }
  };

  useEffect(() => () => { stopCamera(); }, []);

  const switchMode = next => {
    if (next === mode) return;
    recordingRef.current = false;
    framesRef.current = [];
    setRecording(false);
    voterRef.current.reset();
    setLetterLive(null);
    setProgress({ label: null, count: 0 });
    setWordResult(null);
    modeRef.current = next;
    setMode(next);
    if (next === "words" && phase === "running") {
      setEngineNote("Loading the word model...");
      ensureHolistic().then(() => setEngineNote(""));
    }
  };

  const startRecording = async () => {
    if (recordingRef.current || wordBusy || phase !== "running") return;
    if (!holisticRef.current) {
      setEngineNote("Loading the word tracker...");
      await ensureHolistic();
      setEngineNote("");
    }
    framesRef.current = [];
    setRecordCount(0);
    setWordResult(null);
    recordingRef.current = true;
    setRecording(true);
  };

  // ----- sentence
  const finishWord = () => {
    if (!spelling) return;
    setWords(list => [...list, spelling]);
    setSpelling("");
    voterRef.current.reset();
  };

  const addRecognisedWord = () => {
    if (!wordResult || wordResult.failed || wordResult.low_confidence) return;
    setWords(list => [...list, wordText(wordResult.label)]);
  };

  const deleteLast = () => {
    if (mode === "letters" && spelling) setSpelling(text => text.slice(0, -1));
    else setWords(list => list.slice(0, -1));
  };

  const clearAll = () => {
    setWords([]);
    setSpelling("");
    setSentence({ text: "", source: "" });
    voterRef.current.reset();
  };

  const formSentence = async () => {
    const all = spelling ? [...words, spelling] : words;
    if (!all.length) return;
    setSentenceBusy(true);
    setNotice("");
    try {
      const result = await signsToSentence(all, tokenRef.current);
      setSentence({ text: result.sentence, source: result.source });
    } catch (err) {
      setNotice(err instanceof ApiError ? err.message : "The sentence could not be formed. Please try again.");
    } finally {
      setSentenceBusy(false);
    }
  };

  const readAloud = async () => {
    const text = sentence.text || [...words, spelling].filter(Boolean).join(" ");
    if (!text) return;
    const how = await speakText(text, tokenRef.current);
    if (how === "none") setNotice("Your browser cannot read text aloud. Try Chrome or Edge.");
  };

  const statusLabel = { idle: "Camera off", starting: "Starting...", running: "Live", error: "Camera error" }[phase];
  const letterReady = letterLive && letterLive.confidence >= 0.5;
  const sentenceText = sentence.text || [...words, spelling].filter(Boolean).join(" ");

  return (
    <section className="page-content live-page">
      <div className="live-hero">
        <div>
          <span className="eyebrow">COMMUNICATION STUDIO</span>
          <h1>Live Sign Translator</h1>
          <p>Sign a letter or a word. Each sign is checked before it joins your sentence.</p>
        </div>
        <div className="live-hero-stats">
          <span className={`status-pill ${phase === "running" ? "is-live" : ""}`}>{statusLabel}</span>
          {phase === "running" && <span className="fps-pill">{fps} fps</span>}
        </div>
      </div>

      <div className="mode-switch" role="tablist" aria-label="Recognition mode">
        <button role="tab" aria-selected={mode === "letters"} className={mode === "letters" ? "active" : ""} onClick={() => switchMode("letters")}>
          <Hand size={16} /> Letters (A to Z)
        </button>
        <button role="tab" aria-selected={mode === "words"} className={mode === "words" ? "active" : ""} onClick={() => switchMode("words")}>
          <Languages size={16} /> Words ({WORDS.length})
        </button>
      </div>

      <div className="live-grid">
        <div className="panel live-camera">
          <div className="live-screen">
            <video ref={videoRef} className="mirror" playsInline muted />
            <canvas ref={canvasRef} className="mirror live-overlay" />
            {phase !== "running" && (
              <div className="live-placeholder">
                <div className="placeholder-icon"><Camera size={30} /></div>
                <strong>{phase === "starting" ? (engineNote || "Starting your camera...") : "Your camera preview"}</strong>
                <span>Your video stays on this computer. Letters are read in your browser. Only word signs are sent to the server, as hand and body points.</span>
              </div>
            )}
            {phase === "running" && (
              <span className={`hand-chip ${handVisible ? "is-ok" : ""}`}>
                {engineNote || (handVisible ? "Hand detected" : mode === "letters" ? "Show one hand to the camera" : "Show your hands and face")}
              </span>
            )}
            {recording && (
              <div className="record-badge"><span className="record-dot" /> Recording {recordCount} / {FRAMES}</div>
            )}
            {flash && <div className="letter-flash">{flash}</div>}
          </div>

          <div className="live-actions">
            {phase !== "running" ? (
              <button className="primary-button" onClick={startCamera} disabled={phase === "starting"}>
                <Camera size={16} /> {phase === "starting" ? "Starting..." : "Start camera"}
              </button>
            ) : (
              <button className="secondary-button" onClick={stopCamera}><Square size={15} /> Stop camera</button>
            )}
            {mode === "words" && phase === "running" && (
              <button className="primary-button record-button" onClick={startRecording} disabled={recording || wordBusy}>
                <Target size={16} /> {recording ? "Recording..." : "Record a word"}
              </button>
            )}
          </div>
          {error && <p className="error-message" role="alert">{error}</p>}
          {notice && <p className="notice-message">{notice}</p>}
        </div>

        <div className="panel live-results">
          {mode === "letters" ? (
            <div className="result-card">
              <span className="result-kicker">Letter</span>
              <div className={`result-main ${flash ? "pop" : ""}`}>{letterReady ? letterLive.label : "—"}</div>
              <div className="confidence">
                <div className="confidence-bar"><span style={{ width: percent(letterLive?.confidence) }} /></div>
                <small>{letterLive ? `${percent(letterLive.confidence)} sure` : "Waiting for a hand"}</small>
              </div>
              <div className="lock-progress" aria-label="Agreement">
                {Array.from({ length: 5 }).map((_, i) => (
                  <span key={i} className={i < progress.count && progress.label === letterLive?.label ? "on" : ""} />
                ))}
                <small>{progress.count >= 5 ? "Added" : "Hold steady to add"}</small>
              </div>
              {letterLive?.top?.length > 0 && (
                <div className="top-chips">
                  {letterLive.top.map(item => <span key={item.label}>{item.label} {percent(item.confidence)}</span>)}
                </div>
              )}
            </div>
          ) : (
            <div className="result-card">
              <span className="result-kicker">Word</span>
              {wordBusy && <div className="result-main result-busy">Recognising...</div>}
              {!wordBusy && !wordResult && <div className="result-main">—</div>}
              {!wordBusy && wordResult?.failed && <p className="error-message">{wordResult.message}</p>}
              {!wordBusy && wordResult && !wordResult.failed && (
                <>
                  <div className="result-main">{wordResult.low_confidence ? "Not sure" : wordText(wordResult.label)}</div>
                  <div className="confidence">
                    <div className="confidence-bar"><span style={{ width: percent(wordResult.confidence) }} /></div>
                    <small>{percent(wordResult.confidence)} sure</small>
                  </div>
                  {wordResult.top_k?.length > 0 && (
                    <div className="top-chips">
                      {wordResult.top_k.map(item => <span key={item.label}>{wordText(item.label)} {percent(item.confidence)}</span>)}
                    </div>
                  )}
                  {wordResult.low_confidence
                    ? <p className="hint-message">Please sign the word again.</p>
                    : <button className="secondary-button" onClick={addRecognisedWord}><Check size={15} /> Add to sentence</button>}
                  {wordResult.source !== "real" && (
                    <p className="demo-message">Demo result. The word model is not loaded on the server (it needs TensorFlow).</p>
                  )}
                </>
              )}
              {!wordBusy && !wordResult && <p className="hint-message">Press “Record a word”, then sign the whole word in about two seconds.</p>}
            </div>
          )}

          <div className="sentence-box">
            <div className="sentence-head">
              <strong>Your sentence</strong>
              <span>{words.length + (spelling ? 1 : 0)} item{words.length + (spelling ? 1 : 0) === 1 ? "" : "s"}</span>
            </div>
            <div className="sentence-text">
              {sentenceText || <span className="muted-text">Recognised signs appear here.</span>}
            </div>
            {sentence.text && (
              <p className="sentence-source">
                {sentence.source === "real" ? "Formed by the sentence engine (Gemini)." : "Formed by the built-in grammar rules."}
              </p>
            )}
            <div className="sentence-actions">
              {mode === "letters" && <button className="secondary-button" onClick={finishWord} disabled={!spelling}>Finish word</button>}
              <button className="secondary-button" onClick={deleteLast} disabled={!spelling && words.length === 0}>Delete last</button>
              <button className="primary-button small" onClick={formSentence} disabled={sentenceBusy || (!words.length && !spelling)}>
                <Sparkles size={15} /> {sentenceBusy ? "Forming..." : "Form sentence"}
              </button>
              <button className="secondary-button" onClick={readAloud} disabled={!sentenceText}><Volume2 size={15} /> Speak</button>
              <button className="text-button" onClick={clearAll} disabled={!sentenceText}><Trash2 size={14} /> Clear</button>
            </div>
          </div>
        </div>
      </div>

      <div className="live-tips">
        <div className="tip-card"><strong>Good light</strong><span>Face a light source. Avoid bright light behind you.</span></div>
        <div className="tip-card"><strong>Plain background</strong><span>A plain wall behind you helps the camera find your hands.</span></div>
        <div className="tip-card"><strong>Hand in frame</strong><span>Keep your whole hand inside the picture, about an arm's length away.</span></div>
      </div>
    </section>
  );
}
