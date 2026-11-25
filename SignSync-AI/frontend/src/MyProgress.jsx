// My Progress: the learner's real XP, level, streaks, badges and category progress (from /progress/me).
import { useEffect, useState } from "react";
import { Award, Flame, Star, Target, Trophy } from "lucide-react";
import { ApiError, fetchProgress } from "./api";

const percent = value => `${Math.round(Number(value) || 0)}%`;

export default function MyProgress({ token }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchProgress(token)
      .then(result => { if (!cancelled) setData(result); })
      .catch(err => { if (!cancelled) setError(err instanceof ApiError ? err.message : "Your progress could not be loaded."); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [token]);

  const g = data?.gamification;
  const lessons = data?.lessons || [];
  const categories = data?.categories || [];
  const empty = data && g && g.xp === 0 && lessons.every(l => !l.attempts_count);

  return (
    <section className="page-content progress-page">
      <div className="live-hero">
        <div>
          <span className="eyebrow">YOUR JOURNEY</span>
          <h1>My Progress</h1>
          <p>Every sign you practise adds XP. Keep your streak going and unlock badges.</p>
        </div>
      </div>

      {loading && <div className="panel tv-empty"><p>Loading your progress...</p></div>}
      {error && <p className="error-message" role="alert">{error}</p>}

      {g && (
        <>
          <div className="progress-stats">
            <div className="stat-card tone-green"><Star size={20} /><span>Level {g.level}</span><strong>{g.xp} XP</strong></div>
            <div className="stat-card tone-orange"><Flame size={20} /><span>Current streak</span><strong>{g.streak} day{g.streak === 1 ? "" : "s"}</strong></div>
            <div className="stat-card tone-blue"><Trophy size={20} /><span>Best streak</span><strong>{g.best_streak} day{g.best_streak === 1 ? "" : "s"}</strong></div>
            <div className="stat-card tone-purple"><Award size={20} /><span>Badges</span><strong>{g.badges.length}</strong></div>
          </div>

          {!empty && g.xp === 0 && (
            <p className="hint-message">Your progress updates as you practise. Open Live Translator or Learn ISL to start.</p>
          )}
          {empty && (
            <div className="panel tv-empty">
              <div className="tv-empty-icon"><Target size={26} /></div>
              <p><strong>Your progress starts here.</strong> Open Learn ISL or Live Translator, sign your first letter, and your XP, streak and lessons will appear on this page.</p>
            </div>
          )}

          {g.badges.length > 0 && (
            <div className="panel badge-panel">
              <h3>Your badges</h3>
              <div className="badge-row">{g.badges.map(b => <span key={b} className="badge-pill">{b}</span>)}</div>
            </div>
          )}

          {categories.length > 0 && (
            <div className="panel progress-panel">
              <h3>Categories</h3>
              {categories.map(c => (
                <div className="cat-row" key={c.category}>
                  <div className="cat-head"><strong>{c.category}</strong><span>{c.mastered} of {c.total} mastered</span></div>
                  <div className="confidence-bar"><span style={{ width: percent(c.percent) }} /></div>
                </div>
              ))}
            </div>
          )}

          {lessons.length > 0 && (
            <div className="panel progress-panel">
              <h3>Lessons</h3>
              <table className="lesson-table">
                <thead><tr><th>Lesson</th><th>Done</th><th>Best score</th><th>Attempts</th></tr></thead>
                <tbody>
                  {lessons.map(l => (
                    <tr key={l.lesson_id}>
                      <td>{l.title}</td>
                      <td><div className="confidence-bar small"><span style={{ width: percent(l.completion_pct) }} /></div></td>
                      <td>{Math.round(l.best_score)}</td>
                      <td>{l.attempts_count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </section>
  );
}
