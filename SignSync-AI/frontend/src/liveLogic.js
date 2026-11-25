// Decides when a letter is accepted while the learner signs.
// A letter is added only after the same letter is predicted confidently for several frames in a row.
// After it is added, the same letter is ignored until the hand is lowered (no hand seen), so one
// sign does not repeat many times.

export class LetterStabilizer {
  constructor({ needed = 6, minConfidence = 0.6 } = {}) {
    this.needed = needed;
    this.minConfidence = minConfidence;
    this.reset();
  }

  reset() {
    this.candidate = null;
    this.streak = 0;
    this.locked = false;   // true after a letter was accepted, until the hand disappears
  }

  // label: predicted letter or null (no hand, or not confident). Returns the accepted letter or null.
  push(label, confidence) {
    if (label === null || label === undefined || confidence < this.minConfidence) {
      this.candidate = null;
      this.streak = 0;
      if (label === null) this.locked = false;   // hand gone: allow the next sign
      return null;
    }
    if (this.locked && label === this.lockedLabel) return null;
    if (label === this.candidate) {
      this.streak += 1;
    } else {
      this.candidate = label;
      this.streak = 1;
    }
    if (this.streak >= this.needed) {
      this.locked = true;
      this.lockedLabel = label;
      this.candidate = null;
      this.streak = 0;
      return label;
    }
    return null;
  }
}

// Builds the sentence shown to the learner. Letters become a word, words are joined with spaces.
export function sentenceText(words, current) {
  return [...words, current].filter(Boolean).join(" ");
}

// Smooths single-frame predictions. A letter is accepted only when most of the last frames agree,
// so a single wrong frame does not show a wrong letter. After a letter is accepted, nothing is
// accepted again until the hand has been lowered (no hand for a few frames).
export class LetterVoter {
  constructor({ windowSize = 7, needed = 5, minConfidence = 0.5, handGoneFrames = 4 } = {}) {
    this.windowSize = windowSize;
    this.needed = needed;
    this.minConfidence = minConfidence;
    this.handGoneFrames = handGoneFrames;
    this.reset();
  }

  reset() {
    this.votes = [];
    this.lockedLabel = null;
    this.missing = 0;
  }

  // Current leading candidate and how many of the recent frames agree on it.
  status() {
    const counts = new Map();
    for (const v of this.votes) counts.set(v.label, (counts.get(v.label) || 0) + 1);
    let best = null;
    for (const [label, count] of counts) if (!best || count > best.count) best = { label, count };
    return best || { label: null, count: 0 };
  }

  // label/confidence from one frame, or label null when no hand is visible.
  // Returns the accepted letter (once), otherwise null.
  push(label, confidence) {
    if (label === null || label === undefined) {
      this.missing += 1;
      if (this.missing >= this.handGoneFrames) {
        this.lockedLabel = null;
        this.votes = [];
      }
      return null;
    }
    this.missing = 0;
    if (confidence < this.minConfidence) {
      this.votes.push({ label: null, confidence: 0 });
    } else {
      this.votes.push({ label, confidence });
    }
    if (this.votes.length > this.windowSize) this.votes.shift();

    const { label: leader, count } = this.status();
    if (leader && leader !== this.lockedLabel && count >= this.needed) {
      this.lockedLabel = leader;
      this.votes = [];
      return leader;
    }
    return null;
  }
}
