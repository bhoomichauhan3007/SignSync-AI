// Loads MediaPipe Holistic in the browser (pinned version, from the public jsDelivr CDN).
// The same model family produces the landmarks that Member 1's Python code uses.
const HOLISTIC_VERSION = "0.5.1675471629";
const BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@${HOLISTIC_VERSION}`;
let loading = null;

export function loadHolistic() {
  if (window.Holistic) return Promise.resolve(window.Holistic);
  if (loading) return loading;
  loading = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = `${BASE}/holistic.js`;
    script.crossOrigin = "anonymous";
    script.async = true;
    script.onload = () => (window.Holistic ? resolve(window.Holistic) : reject(new Error("Recognition engine did not start.")));
    script.onerror = () => {
      loading = null;
      reject(new Error("Could not load the recognition engine. Check your internet connection and refresh the page."));
    };
    document.head.appendChild(script);
  });
  return loading;
}

export function createHolistic(Holistic) {
  const holistic = new Holistic({ locateFile: file => `${BASE}/${file}` });
  holistic.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    enableSegmentation: false,
    refineFaceLandmarks: false,   // keeps 468 face points, as in the 1662 layout
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  return holistic;
}

// MediaPipe Hands: lighter than Holistic, used for letters. It is the same hand model Member 1 used
// to extract the alphabet features.
const HANDS_VERSION = "0.4.1675469240";
const HANDS_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/hands@${HANDS_VERSION}`;
let handsLoading = null;

export function loadHands() {
  if (window.Hands) return Promise.resolve(window.Hands);
  if (handsLoading) return handsLoading;
  handsLoading = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = `${HANDS_BASE}/hands.js`;
    script.crossOrigin = "anonymous";
    script.async = true;
    script.onload = () => (window.Hands ? resolve(window.Hands) : reject(new Error("Hand recognition did not start.")));
    script.onerror = () => {
      handsLoading = null;
      reject(new Error("Could not load the hand recognition engine. Check your internet connection and refresh the page."));
    };
    document.head.appendChild(script);
  });
  return handsLoading;
}

export function createHands(Hands) {
  const hands = new Hands({ locateFile: file => `${HANDS_BASE}/${file}` });
  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.6,
    minTrackingConfidence: 0.5,
  });
  return hands;
}
