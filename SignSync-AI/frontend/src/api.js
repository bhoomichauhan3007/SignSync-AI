// Client for the SignSync backend. The address comes from VITE_API_BASE_URL (see .env).
const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

// FastAPI returns validation problems as a list; turn them into one readable sentence.
function readableDetail(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map(item => (item && item.msg ? item.msg.replace(/^Value error, /, "") : ""))
      .filter(Boolean)
      .join(". ") || "Please check the form and try again.";
  }
  return "";
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, options);
  } catch {
    throw new ApiError("Cannot reach the SignSync server. Please make sure it is running.", 0);
  }
  if (!response.ok) {
    let detail = "";
    try {
      detail = readableDetail((await response.json()).detail);
    } catch {
      /* the body was not JSON */
    }
    if (response.status === 401) {
      const message = path === "/auth/login"
        ? "Incorrect email or password."
        : "Your session has ended. Please sign in again.";
      throw new ApiError(message, 401);
    }
    if (response.status === 409) {
      throw new ApiError(detail || "This email is already registered. Try signing in instead.", 409);
    }
    throw new ApiError(detail || `Something went wrong (error ${response.status}). Please try again.`, response.status);
  }
  return response.json();
}

const jsonHeaders = { "Content-Type": "application/json" };
const authHeaders = token => ({ Authorization: `Bearer ${token}` });

export function login(email, password) {
  return request("/auth/login", {
    method: "POST",
    headers: jsonHeaders,
    body: JSON.stringify({ email, password }),
  });
}

export function register({ name, email, password, profile }) {
  return request("/auth/register", {
    method: "POST",
    headers: jsonHeaders,
    body: JSON.stringify({ name, email, password, profile }),
  });
}

export function fetchMe(token) {
  return request("/users/me", { headers: authHeaders(token) });
}

export function askTutor(question, token, conversationId = null) {
  return request("/assistant/ask", {
    method: "POST",
    headers: { ...jsonHeaders, ...authHeaders(token) },
    body: JSON.stringify({ question, conversation_id: conversationId }),
  });
}

// Sends one sign to the recognition model.
// mode "static": 63 hand values (one letter). mode "dynamic": 30 frames of 1662 values (one word).
export function predictSign(landmarks, mode, token) {
  return request("/predict/sign", {
    method: "POST",
    headers: { ...jsonHeaders, ...authHeaders(token) },
    body: JSON.stringify({ landmarks, mode }),
  });
}

// Forms a sentence from recognised words (Member 3: Gemini when configured, rule-based otherwise).
export function signsToSentence(words, token) {
  return request("/translate/signs-to-sentence", {
    method: "POST",
    headers: { ...jsonHeaders, ...authHeaders(token) },
    body: JSON.stringify({ words }),
  });
}

// Turns text into signs: a gloss list with the sign picture or clip for each item.
export function textToSign(text, token) {
  return request("/translate/text-to-sign", {
    method: "POST",
    headers: { ...jsonHeaders, ...authHeaders(token) },
    body: JSON.stringify({ text }),
  });
}

// Returns an audio Blob when the server voice (Member 3, gTTS) is installed, otherwise null.
export async function textToSpeechBlob(text, token) {
  let response;
  try {
    response = await fetch(`${API_BASE}/translate/text-to-speech`, {
      method: "POST",
      headers: { ...jsonHeaders, ...authHeaders(token) },
      body: JSON.stringify({ text, language: "en" }),
    });
  } catch {
    return null;
  }
  if (!response.ok) return null;
  const type = response.headers.get("content-type") || "";
  if (!type.startsWith("audio/")) return null;
  return response.blob();
}

// Sends a recorded voice clip to the server. Member 3's Whisper turns it into text, then into signs.
export async function speechToSign(audioBlob, token, language = "en") {
  const form = new FormData();
  form.append("audio", audioBlob, "speech.webm");
  form.append("language", language);
  return request("/translate/speech-to-sign", {
    method: "POST",
    headers: authHeaders(token),
    body: form,
  });
}

// Learner progress: XP, level, streaks, badges, lesson and category progress (Member 5).
export function fetchProgress(token) {
  return request("/progress/me", { headers: authHeaders(token) });
}
