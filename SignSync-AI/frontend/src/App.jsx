
import { useState, useRef, useEffect, Component } from "react";
import { askTutor, ApiError, fetchMe, fetchProgress } from "./api";
import AuthScreen from "./AuthScreen";
import LiveTranslator from "./LiveTranslator";
import TextVoice from "./TextVoice";
import MyProgress from "./MyProgress";
import {
  Home, Camera, BookOpen, BrainCircuit, ChartNoAxesCombined,
  Hand, Languages, Settings, CircleHelp, Bell, Search,
  ChevronRight, ChevronDown, Sparkles, Volume2, ShieldCheck,
  Clock3, Flame, Star, Target, TrendingUp, Check, X,
  Send, Trash2, Menu, Play, Square, MessageCircle,
  Lock,
} from "lucide-react";
import { ALPHABET, WORDS, VIDEO_WORDS, PICTURE_EXTENSIONS, isVideo, signPictureUrl } from "./signs";
import { speakWithBrowser } from "./speech";
import "./App.css";

const navigation = [
  { label: "Overview", icon: Home },
  { label: "Live Translator", icon: Camera },
  { label: "Text & Voice", icon: MessageCircle },
  { label: "Learn ISL", icon: BookOpen },
  { label: "AI Tutor", icon: BrainCircuit },
  { label: "My Progress", icon: ChartNoAxesCombined }
];

const suggestions = [
  "How do I start learning ISL?",
  "Explain hand shapes",
  "Give me a practice plan"
];

function AppContent() {
  const [activePage, setActivePage] = useState("Overview");
  const [mobileMenu, setMobileMenu] = useState(false);

  const [cameraOn, setCameraOn] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  const [translatedText, setTranslatedText] = useState("");
  const [history, setHistory] = useState([]);

  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text: "Namaste! 👋 Welcome to SignSync AI Tutor. Ask me anything about Indian Sign Language, hand shapes, vocabulary or learning plans!"
    }
  ]);
  const [typing, setTyping] = useState(false);
  const [token, setToken] = useState(() => localStorage.getItem("signsync_token") || "");
  const [user, setUser] = useState(() => {
    try { return JSON.parse(localStorage.getItem("signsync_user") || "null"); } catch { return null; }
  });
  const firstName = (user?.name || "there").split(" ")[0];
  const chatEndRef = useRef(null);
  const [progress, setProgress] = useState(null);
  useEffect(() => {
    if (!token) return;
    fetchProgress(token).then(setProgress).catch(() => setProgress(null));
  }, [token, activePage]);
  const lessonsDone = progress ? progress.lessons.filter(l => l.completion_pct >= 100).length : 0;
  const lessonsStarted = progress ? progress.lessons.filter(l => l.attempts_count > 0).length : 0;
  const streak = progress?.gamification?.streak ?? 0;
  const xp = progress?.gamification?.xp ?? 0;

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, typing]);

  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach(track => track.stop());
      window.speechSynthesis?.cancel();
    };
  }, []);

  const sendQuestion = async (value = question) => {
    const text = value.trim();
    if (!text || typing) return;

    if (!token) return; // the input is disabled until signed in, so nothing extra appears in the chat

    setMessages(old => [...old, { role: "user", text }]);
    setQuestion("");

    setTyping(true);
    try {
      const reply = await askTutor(text, token);
      setMessages(old => [...old, { role: "assistant", text: reply.answer, sources: reply.sources, mode: reply.mode }]);
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) signOut();
      setMessages(old => [...old, { role: "assistant", text: err.message }]);
    } finally {
      setTyping(false);
    }
  };

  const handleAuthenticated = (newToken, newUser) => {
    localStorage.setItem("signsync_token", newToken);
    localStorage.setItem("signsync_user", JSON.stringify(newUser));
    setToken(newToken);
    setUser(newUser);
  };

  const signOut = () => {
    localStorage.removeItem("signsync_token");
    localStorage.removeItem("signsync_user");
    setToken("");
    setUser(null);
  };

  // When the page opens with a saved session, confirm it is still valid with the server.
  useEffect(() => {
    if (!token) return;
    fetchMe(token)
      .then(setUser)
      .catch(err => { if (err instanceof ApiError && err.status === 401) signOut(); });
  }, [token]);

  const startCamera = async () => {
    try {
      setCameraError("");
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setCameraOn(true);
    } catch {
      setCameraError("Please allow camera permission in your browser settings.");
    }
  };

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach(track => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraOn(false);
  };

  const speakText = () => {
    if (!translatedText.trim()) return;
    window.speechSynthesis.cancel();
    const speech = new SpeechSynthesisUtterance(translatedText);
    speech.lang = "en-IN";
    window.speechSynthesis.speak(speech);
  };

  const saveTranslation = () => {
    if (!translatedText.trim()) return;
    setHistory(old => [
      { id: Date.now(), text: translatedText, time: new Date().toLocaleTimeString() },
      ...old
    ]);
  };

  const renderPage = () => {
    if (activePage === "AI Tutor") {
      return (
        <section className="page-content">
          <div className="page-heading">
            <div>
              <span className="eyebrow">YOUR LEARNING COMPANION</span>
              <h1>Meet your AI Tutor</h1>
              <p>Ask questions and explore sign language concepts.</p>
            </div>
            <span className="status-pill">
              <span className="status-dot"/> Connected
            </span>
          </div>

          <div className="panel tutor-panel">
            <div className="tutor-welcome">
              <div className="tutor-avatar">
                <BrainCircuit size={30}/>
              </div>
              <span className="eyebrow">SIGNSYNC AI ASSISTANT</span>
              <h2>Namaste! How can I help you learn today?</h2>
              <p>Ask me about ISL, hand shapes, signs and learning tips.</p>
            </div>

            <div className="suggestion-chips">
              {suggestions.map((item, index) => (
                <button
                  key={index}
                  onClick={() => sendQuestion(item)}
                  disabled={typing || !token}
                >
                  {item}
                </button>
              ))}
            </div>

            <div className="chat-messages">
              {messages.map((msg, index) => (
                <div
                  key={index}
                  className={`chat-message ${msg.role === "user" ? "user-message" : "assistant-message"}`}
                >
                  <div className="message-avatar">
                    {msg.role === "user" ? "L" : <BrainCircuit size={17}/>}
                  </div>
                  <div className="message-content">
                    <span>{msg.role === "user" ? "You" : "SignSync AI Tutor"}</span>
                    <div className="answer-text">{renderAnswer(msg.text)}</div>
                    {msg.mode && (
                      <span className={`answer-pill mode-${msg.mode}`}>
                        {msg.mode === "llm" ? "Answered from the sources"
                          : msg.mode === "extractive" ? "From the sources (no rewrite)"
                          : msg.mode === "refused" ? "Not in the knowledge base"
                          : "Knowledge base not ready"}
                      </span>
                    )}
                    {msg.sources?.some(src => src.status !== "verified") && msg.mode !== "refused" && (
                      <div className="unverified-banner">
                        Some of this material has not been checked by a qualified ISL teacher yet.
                      </div>
                    )}
                    {msg.sources?.length > 0 && (
                      <div className="source-list">
                        {groupSources(msg.sources).map((src, i) => (
                          <div className="source-card" key={i}>
                            <span className="cite">{src.numbers.map(n => `S${n}`).join(", ")}</span>
                            <div>
                              <strong>{cleanTitle(src.title)}</strong>
                              <small>{src.page ? `Page ${src.page}` : "Document"}</small>
                            </div>
                            <span className={`src-status src-${src.status}`}>
                              {src.status === "verified" ? "Verified" : src.status === "sample" ? "Sample" : "Not yet checked"}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {typing && (
                <div className="chat-message assistant-message">
                  <div className="message-avatar"><BrainCircuit size={17}/></div>
                  <div className="message-content">
                    <span>SignSync AI Tutor</span>
                    <p className="typing-text">Thinking...</p>
                  </div>
                </div>
              )}
              <div ref={chatEndRef}/>
            </div>

            <form
              className="chat-input"
              onSubmit={e => {
                e.preventDefault();
                sendQuestion();
              }}
            >
              <input
                value={question}
                onChange={e => setQuestion(e.target.value)}
                placeholder="Ask your sign language question..."
                disabled={typing || !token}
              />
              <button
                type="submit"
                className="primary-button"
                disabled={!token || !question.trim() || typing}
              >
                <Send size={17}/> Send
              </button>
            </form>

            <div className="tutor-footer">
              <span><Sparkles size={13}/> Answers come from the ISL knowledge base and show their sources</span>
              <button
                onClick={() => setMessages([{
                  role: "assistant",
                  text: "Namaste! 👋 Welcome back. What would you like to learn today?"
                }])}
              >
                <Trash2 size={13}/> Clear chat
              </button>
            </div>
          </div>
        </section>
      );
    }

    if (activePage === "Text & Voice") {
      return <TextVoice token={token} />;
    }

    if (activePage === "Live Translator") {
      return <LiveTranslator token={token} />;
    }

    if (activePage === "Learn ISL") {
      return (
        <section className="page-content">
          <div className="page-heading">
            <div>
              <span className="eyebrow">YOUR LEARNING SPACE</span>
              <h1>Learn Indian Sign Language</h1>
              <p>Small steps, meaningful connections. Keep learning at your own pace.</p>
            </div>
          </div>
          <div className="learning-banner">
            <div>
              <span className="eyebrow">START YOUR JOURNEY</span>
              <h2>Every sign brings us closer.</h2>
              <p>Explore lessons, practise new signs and build your confidence.</p>
              <button className="primary-button" onClick={() => setActivePage("Live Translator")}>Practice a sign <ChevronRight size={16}/></button>
            </div>
            <div className="learning-art"><Hand size={80}/></div>
          </div>
          <h2 className="section-title">Sign gallery</h2>
          <SignGallery />
          <h2 className="section-title">Your lessons</h2>
          <div className="lesson-grid">
            {["Basic Greetings", "Everyday Words", "Numbers & Counting"].map((lesson, i) => (
              <div className="panel lesson-card" key={lesson}>
                <div className="lesson-icon"><Hand size={22}/></div>
                <span className="eyebrow">{i === 1 ? "VOCABULARY" : "BEGINNER"}</span>
                <h3>{lesson}</h3>
                <p className="muted-text">{[8,12,10][i]} min lesson</p>
                <button className="text-button" onClick={() => setActivePage("Live Translator")}>Practice lesson <ChevronRight size={15}/></button>
              </div>
            ))}
          </div>
        </section>
      );
    }

    if (activePage === "My Progress") {
      return <MyProgress token={token} />;
    }

    return (
      <section className="page-content">
        <div className="welcome-row">
          <div>
            <span className="eyebrow">YOUR LEARNING JOURNEY CONTINUES</span>
            <h1>Welcome back, {firstName} <span className="wave">✦</span></h1>
            <p>Ready to make meaningful connections through sign language?</p>
          </div>
        </div>
        <div className="hero-card">
          <div className="hero-copy">
            <span className="hero-tag"><Sparkles size={14}/> YOUR DAILY INSPIRATION</span>
            <h2>Every hand tells<br/>a beautiful story.</h2>
            <p>Learn a new sign today. Make communication more inclusive, one gesture at a time.</p>
            <button className="hero-button" onClick={() => setActivePage("Learn ISL")}>Explore lessons <ChevronRight size={17}/></button>
          </div>
          <div className="hero-visual"><div className="hero-circle circle-one"/><div className="hero-circle circle-two"/><div className="hero-hand"><Hand size={90}/></div></div>
        </div>
        <div className="stats-grid">
          <Stat icon={BookOpen} label="Lessons completed" value={String(lessonsDone)} note={lessonsDone ? "Well done" : "Start your first lesson"}/>
          <Stat icon={Flame} label="Learning streak" value={`${streak} day${streak === 1 ? "" : "s"}`} note={streak ? "Keep going" : "Practise today"}/>
          <Stat icon={Star} label="Practice points" value={`${xp} XP`} note={xp ? "Earned so far" : "Earn XP by practising"}/>
          <Stat icon={Target} label="Lessons started" value={`${lessonsStarted} / 5`} note={lessonsStarted >= 5 ? "Target reached" : "Target: 5 lessons"}/>
        </div>
        <div className="section-heading">
          <div><span className="eyebrow">KEEP YOUR MOMENTUM</span><h2>Pick up where you left off</h2></div>
          <button className="text-button" onClick={() => setActivePage("Learn ISL")}>View all lessons <ChevronRight size={15}/></button>
        </div>
        <div className="lesson-grid">
          {["Basic Greetings", "Everyday Words", "Numbers & Counting"].map((lesson, i) => (
            <div className="panel lesson-card" key={lesson}>
              <div className="lesson-icon"><Hand size={22}/></div>
              <h3>{lesson}</h3>
              <div className="progress-track"><div style={{width: `${[100,65,30][i]}%`}}/></div>
              <button className="text-button" onClick={() => setActivePage("Learn ISL")}>Continue learning <ChevronRight size={15}/></button>
            </div>
          ))}
        </div>
        <div className="panel quick-panel">
          <h2>Quick access</h2>
          <button className="text-button" onClick={() => setActivePage("Live Translator")}>Open Live Translator <ChevronRight size={15}/></button>
          <button className="text-button" onClick={() => setActivePage("AI Tutor")}>Talk to AI Tutor <ChevronRight size={15}/></button>
        </div>
      </section>
    );
  };

  if (!token) {
    return <AuthScreen onAuthenticated={handleAuthenticated} />;
  }

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileMenu ? "sidebar-open" : ""}`}>
        <div className="brand">
          <div className="brand-mark"><Hand size={23}/></div>
          <div><strong>SignSync<span>AI</span></strong><small>COMMUNICATION FOR ALL</small></div>
          <button className="mobile-close" onClick={() => setMobileMenu(false)}><X size={19}/></button>
        </div>
        <div className="workspace-label">WORKSPACE</div>
        <nav className="nav-list">
          {navigation.map(item => {
            const Icon = item.icon;
            return (
              <button key={item.label} className={`nav-item ${activePage === item.label ? "active" : ""}`} onClick={() => {setActivePage(item.label);setMobileMenu(false);}}>
                <Icon size={19}/><span>{item.label}</span>
                {item.label === "Live Translator" && <span className="nav-new">AI</span>}
              </button>
            );
          })}
        </nav>
        <div className="sidebar-bottom">
          <div className="help-card">
            <CircleHelp size={19}/>
            <strong>Need a little help?</strong>
            <p>We're here to make your learning journey easier.</p>
            <button onClick={() => setActivePage("AI Tutor")}>Visit AI Tutor <ChevronRight size={14}/></button>
          </div>
          <button className="nav-item settings-link"><Settings size={19}/><span>Settings</span></button>
          <div className="profile-mini"><div className="profile-avatar">{(user?.name || "L").charAt(0).toUpperCase()}</div><div><strong>{user?.name || "Learner"}</strong><small>{user?.email || "My learning profile"}</small></div><button type="button" className="signout-button" onClick={signOut}>Sign out</button></div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <button className="mobile-menu" onClick={() => setMobileMenu(true)}>☰</button>
          <div className="breadcrumb">Workspace <ChevronRight size={14}/> <strong>{activePage}</strong></div>
          <div className="topbar-actions">
            <div className="search-box"><Search size={16}/><input placeholder="Search anything..."/></div>
            <button className="icon-button"><Bell size={19}/></button>
            <div className="top-avatar">L</div>
          </div>
        </header>
        {renderPage()}
        <footer className="footer">
          <span>© 2026 SignSync AI</span>
          <span>Making communication more inclusive ♥</span>
          <span>Built with care</span>
        </footer>
      </main>
    </div>
  );
}

// Splits an answer into paragraphs and turns [S1] markers into small badges.
function renderAnswer(text) {
  return (text || "").split(/\n\s*\n/).map((para, i) => (
    <p key={i}>
      {para.split(/(\[S\d+\])/g).map((part, j) => {
        const match = part.match(/^\[S(\d+)\]$/);
        return match ? <sup key={j} className="cite">S{match[1]}</sup> : <span key={j}>{part}</span>;
      })}
    </p>
  ));
}

// Groups citations by document file, so one document appears once with all its S numbers.
function groupSources(sources) {
  const groups = new Map();
  sources.forEach((src, i) => {
    const key = src.source || src.title;
    if (!groups.has(key)) groups.set(key, { ...src, numbers: [] });
    groups.get(key).numbers.push(i + 1);
  });
  return [...groups.values()];
}

// Shows a document title without internal labels such as "SAMPLE TEST DOCUMENT - ...".
function cleanTitle(title) {
  const t = (title || "").replace(/^SAMPLE TEST DOCUMENT\s*-\s*/i, "").replace(/\s*\(UNVERIFIED[^)]*\)/i, "").trim();
  return t.charAt(0).toUpperCase() + t.slice(1);
}

// Shows an error message instead of a blank white page if a screen crashes.
class PageErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error) {
    return { error };
  }
  render() {
    if (this.state.error) {
      return (
        <div className="panel" style={{ margin: 24, padding: 20 }}>
          <h3>This page could not be shown</h3>
          <p>Please send this message to the developer:</p>
          <pre style={{ whiteSpace: "pre-wrap" }}>{String(this.state.error?.message || this.state.error)}</pre>
        </div>
      );
    }
    return this.props.children;
  }
}

function SignTile({ kind, item }) {
  // Video dictionary tiles only have a video; other tiles try video, photo, then picture card.
  const extensions = kind === "videos" ? ["mp4"] : PICTURE_EXTENSIONS;
  const [attempt, setAttempt] = useState(0);
  const [missing, setMissing] = useState(false);
  const label = kind === "alphabet" || kind === "videos" ? item.text : item.text.charAt(0);
  const onPictureError = () => {
    if (attempt + 1 < extensions.length) setAttempt(attempt + 1);
    else setMissing(true);
  };
  const src = signPictureUrl(kind, item.gloss, attempt);
  return (
    <figure className="sign-tile" onClick={() => speakWithBrowser(item.text)} title="Click to hear it">
      <div className="sign-picture">
        {missing ? (
          <div className="sign-placeholder">
            <span>{label}</span>
            <small>{kind === "videos" ? "Video coming soon" : "Picture coming soon"}</small>
          </div>
        ) : isVideo(extensions[attempt]) ? (
          <video key={src} src={src} aria-label={`Sign video for ${item.text}`} autoPlay loop muted playsInline preload="metadata" onError={onPictureError} />
        ) : (
          <img src={src} alt={`Sign for ${item.text}`} loading="lazy" onError={onPictureError} />
        )}
      </div>
      <figcaption>{item.text}</figcaption>
    </figure>
  );
}

function SignGallery() {
  const [tab, setTab] = useState("alphabet");
  const items = tab === "alphabet"
    ? ALPHABET.map(letter => ({ gloss: letter, text: letter }))
    : VIDEO_WORDS.map(w => ({ gloss: w.slug, text: w.text }));
  return (
    <div className="panel gallery">
      <div className="gallery-tabs" role="tablist" aria-label="Sign type">
        <button role="tab" aria-selected={tab === "alphabet"} className={tab === "alphabet" ? "active" : ""} onClick={() => setTab("alphabet")}>
          Alphabet (A to Z)
        </button>
        <button role="tab" aria-selected={tab === "videos"} className={tab === "videos" ? "active" : ""} onClick={() => setTab("videos")}>
          Video dictionary ({VIDEO_WORDS.length})
        </button>
      </div>
      <p className="muted-text gallery-hint">
        Each tile shows the sign. The Video dictionary shows the videos you have added in public/signs/videos.
      </p>
      <div className="sign-grid">
        {items.map(item => <SignTile key={`${tab}-${item.gloss}`} kind={tab} item={item} />)}
      </div>
    </div>
  );
}

function Stat({ icon: Icon, label, value, note }) {
  return (
    <div className="panel stat-card">
      <div className="stat-top">
        <span className="stat-icon"><Icon size={19}/></span>
        <span className="stat-note"><TrendingUp size={13}/>{note}</span>
      </div>
      <strong>{value}</strong>
      <span className="stat-label">{label}</span>
    </div>
  );
}

export default function App() {
  return (
    <PageErrorBoundary>
      <AppContent />
    </PageErrorBoundary>
  );
}
