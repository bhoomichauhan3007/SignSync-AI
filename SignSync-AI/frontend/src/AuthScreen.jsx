// First screen of the app: sign in, or create an account with a short profile.
import { useState } from "react";
import { ArrowLeft, ArrowRight, Check, Hand, Lock, Mail, ShieldCheck, User } from "lucide-react";
import { login, register } from "./api";

const WHO_OPTIONS = [
  ["learner", "I am learning ISL"],
  ["parent", "Parent or family member of a deaf person"],
  ["teacher", "Teacher or educator"],
  ["friend", "Friend, colleague or neighbour"],
  ["professional", "Health, service or public-facing work"],
  ["other", "Something else"],
];
const AGE_OPTIONS = [
  ["under_18", "Under 18"],
  ["18_25", "18 to 25"],
  ["26_40", "26 to 40"],
  ["41_60", "41 to 60"],
  ["over_60", "Over 60"],
];
const EXPERIENCE_OPTIONS = [
  ["never", "I have never learned ISL"],
  ["few_signs", "I know a few signs"],
  ["basics", "I can sign basic sentences"],
  ["regular", "I use ISL regularly"],
];
const GOAL_OPTIONS = [
  ["fun_culture", "Learn for interest or culture"],
  ["school_college", "School or college"],
  ["communicate", "Talk with someone who is deaf"],
  ["work_service", "Work or customer service"],
  ["other", "Something else"],
];

const EMPTY_FORM = {
  name: "", email: "", password: "", confirm: "",
  who: "", age_group: "", experience: "", goal: "", problem: "", consent: false,
};

function ChoiceGroup({ legend, name, options, value, onChange, required = true }) {
  return (
    <fieldset className="auth-fieldset">
      <legend>{legend}{required && <span className="required"> *</span>}</legend>
      <div className="choice-grid">
        {options.map(([optionValue, label]) => (
          <label key={optionValue} className={`choice ${value === optionValue ? "selected" : ""}`}>
            <input
              type="radio"
              name={name}
              value={optionValue}
              checked={value === optionValue}
              onChange={() => onChange(name, optionValue)}
            />
            <span>{label}</span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}

export default function AuthScreen({ onAuthenticated }) {
  const [mode, setMode] = useState("signin");   // "signin" | "signup"
  const [step, setStep] = useState(1);          // sign-up only: 1 = account, 2 = about you
  const [form, setForm] = useState(EMPTY_FORM);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const setField = (field, value) => setForm(current => ({ ...current, [field]: value }));
  const textField = field => e => setField(field, e.target.value);

  function switchMode(next) {
    setMode(next);
    setStep(1);
    setError("");
  }

  function checkAccountStep() {
    if (!form.name.trim()) return "Please enter your name.";
    if (!/^\S+@\S+\.\S+$/.test(form.email.trim())) return "Please enter a valid email address.";
    if (form.password.length < 8) return "Your password must be at least 8 characters long.";
    if (form.password !== form.confirm) return "The two passwords do not match.";
    return "";
  }

  function checkProfileStep() {
    if (!form.who || !form.age_group || !form.experience || !form.goal) {
      return "Please answer every question marked with *.";
    }
    if (!form.consent) return "Please accept the privacy notice to create your account.";
    return "";
  }

  async function handleSignIn(event) {
    event.preventDefault();
    setError("");
    setBusy(true);
    try {
      const data = await login(form.email.trim(), form.password);
      onAuthenticated(data.access_token, data.user);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  function goToProfileStep(event) {
    event.preventDefault();
    const message = checkAccountStep();
    if (message) {
      setError(message);
      return;
    }
    setError("");
    setStep(2);
  }

  async function handleCreateAccount(event) {
    event.preventDefault();
    const message = checkProfileStep();
    if (message) {
      setError(message);
      return;
    }
    setError("");
    setBusy(true);
    try {
      const data = await register({
        name: form.name.trim(),
        email: form.email.trim(),
        password: form.password,
        profile: {
          who: form.who,
          age_group: form.age_group,
          experience: form.experience,
          goal: form.goal,
          problem: form.problem.trim(),
          consent: true,
        },
      });
      onAuthenticated(data.access_token, data.user);
    } catch (err) {
      setError(err.message);
      if (err.status === 409) setStep(1);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="auth-shell">
      <aside className="auth-brand">
        <div className="brand-mark"><Hand size={26} /></div>
        <h1>SignSync<span>AI</span></h1>
        <p className="auth-tagline">
          Communication for all. Learn Indian Sign Language, turn signs into text and speech,
          and ask our AI tutor your questions.
        </p>
        <ul className="auth-points">
          <li><Check size={16} /> Learn the alphabet and everyday words</li>
          <li><Check size={16} /> Ask the AI tutor. Every answer shows its source</li>
          <li><Check size={16} /> Follow your own progress as you learn</li>
        </ul>
        <div className="auth-privacy"><ShieldCheck size={16} /> Your password is stored only in encrypted form.</div>
      </aside>

      <main className="auth-main">
        <div className="auth-card">
          <div className="auth-tabs" role="tablist">
            <button role="tab" aria-selected={mode === "signin"} className={mode === "signin" ? "active" : ""} onClick={() => switchMode("signin")}>
              Sign in
            </button>
            <button role="tab" aria-selected={mode === "signup"} className={mode === "signup" ? "active" : ""} onClick={() => switchMode("signup")}>
              Create account
            </button>
          </div>

          {mode === "signin" && (
            <form className="auth-form" onSubmit={handleSignIn}>
              <h2>Welcome back</h2>
              <p className="auth-lead">Sign in to continue your learning.</p>
              <label className="field">
                <span>Email</span>
                <div className="field-input"><Mail size={16} /><input type="email" value={form.email} onChange={textField("email")} placeholder="you@example.com" autoComplete="username" required /></div>
              </label>
              <label className="field">
                <span>Password</span>
                <div className="field-input"><Lock size={16} /><input type="password" value={form.password} onChange={textField("password")} placeholder="Your password" autoComplete="current-password" required /></div>
              </label>
              <button type="submit" className="auth-submit" disabled={busy}>{busy ? "Signing in..." : "Sign in"}</button>
              <p className="auth-switch">New here? <button type="button" onClick={() => switchMode("signup")}>Create an account</button></p>
            </form>
          )}

          {mode === "signup" && step === 1 && (
            <form className="auth-form" onSubmit={goToProfileStep}>
              <p className="step-label">Step 1 of 2 · Your account</p>
              <h2>Create your account</h2>
              <label className="field">
                <span>Full name *</span>
                <div className="field-input"><User size={16} /><input value={form.name} onChange={textField("name")} placeholder="Your name" autoComplete="name" required /></div>
              </label>
              <label className="field">
                <span>Email *</span>
                <div className="field-input"><Mail size={16} /><input type="email" value={form.email} onChange={textField("email")} placeholder="you@example.com" autoComplete="email" required /></div>
              </label>
              <div className="field-row">
                <label className="field">
                  <span>Password *</span>
                  <div className="field-input"><Lock size={16} /><input type="password" value={form.password} onChange={textField("password")} placeholder="At least 8 characters" autoComplete="new-password" required /></div>
                </label>
                <label className="field">
                  <span>Confirm password *</span>
                  <div className="field-input"><Lock size={16} /><input type="password" value={form.confirm} onChange={textField("confirm")} placeholder="Type it again" autoComplete="new-password" required /></div>
                </label>
              </div>
              <button type="submit" className="auth-submit">Continue <ArrowRight size={16} /></button>
            </form>
          )}

          {mode === "signup" && step === 2 && (
            <form className="auth-form" onSubmit={handleCreateAccount}>
              <p className="step-label">Step 2 of 2 · About you</p>
              <h2>Tell us about you</h2>
              <p className="auth-lead">This helps us suggest the right lessons. You can skip the last box.</p>
              <ChoiceGroup legend="Which describes you best?" name="who" options={WHO_OPTIONS} value={form.who} onChange={setField} />
              <ChoiceGroup legend="Your age group" name="age_group" options={AGE_OPTIONS} value={form.age_group} onChange={setField} />
              <ChoiceGroup legend="How much ISL do you already know?" name="experience" options={EXPERIENCE_OPTIONS} value={form.experience} onChange={setField} />
              <ChoiceGroup legend="What is your main goal?" name="goal" options={GOAL_OPTIONS} value={form.goal} onChange={setField} />
              <label className="field">
                <span>Tell us what you need help with (optional)</span>
                <textarea
                  rows={3}
                  maxLength={1000}
                  value={form.problem}
                  onChange={textField("problem")}
                  placeholder="For example: I want to talk with my deaf neighbour, or I need to learn signs for my exam."
                />
                <small className="counter">{form.problem.length} / 1000</small>
              </label>
              <label className="consent">
                <input type="checkbox" checked={form.consent} onChange={e => setField("consent", e.target.checked)} />
                <span>I agree that SignSync AI stores these details to personalise my learning.</span>
              </label>
              <div className="auth-actions">
                <button type="button" className="auth-back" onClick={() => setStep(1)}><ArrowLeft size={16} /> Back</button>
                <button type="submit" className="auth-submit" disabled={busy}>{busy ? "Creating account..." : "Create account"}</button>
              </div>
            </form>
          )}

          {error && <p className="auth-error" role="alert">{error}</p>}
        </div>
      </main>
    </div>
  );
}
