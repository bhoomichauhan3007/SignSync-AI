// Node tests for the letter voting and hand-feature logic used by the Live Translator.
const fs = require("fs");
const path = require("path");
const assert = require("assert");
const ts = require("/opt/npm-tools/node_modules/typescript");
function load(name) {
  const src = fs.readFileSync(path.join(__dirname, "../src", name), "utf8");
  const out = ts.transpileModule(src, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 } }).outputText;
  const m = { exports: {} };
  new Function("module", "exports", "require", out)(m, m.exports, require);
  return m.exports;
}
const logic = load("liveLogic.js");
const hf = load("holisticFeatures.js");

// Voter: needs 5 of the last 7 frames to agree, and accepts a letter once.
const v = new logic.LetterVoter({ windowSize: 7, needed: 5, minConfidence: 0.5, handGoneFrames: 4 });
let accepted = [];
const feed = (label, conf, n) => { for (let i = 0; i < n; i++) { const r = v.push(label, conf); if (r) accepted.push(r); } };
feed("A", 0.9, 4); assert.deepStrictEqual(accepted, [], "not enough agreement yet");
feed("A", 0.9, 2); assert.deepStrictEqual(accepted, ["A"], "A accepted after 5 agreeing frames");
feed("A", 0.9, 20); assert.deepStrictEqual(accepted, ["A"], "A not repeated while the hand stays");
// one wrong frame in the middle does not break a stable letter
v.reset(); accepted = [];
feed("B", 0.9, 2); feed("C", 0.9, 1); feed("B", 0.9, 3);
assert.deepStrictEqual(accepted, ["B"], "a single outlier frame is ignored");
// low confidence frames do not count
v.reset(); accepted = [];
feed("D", 0.2, 10); assert.deepStrictEqual(accepted, [], "low confidence never accepted");
// hand lowered for a few frames unlocks the same letter again
v.reset(); accepted = [];
feed("E", 0.9, 5); assert.deepStrictEqual(accepted, ["E"]);
for (let i = 0; i < 4; i++) v.push(null, 0);
feed("E", 0.9, 5); assert.deepStrictEqual(accepted, ["E", "E"], "E can be signed again after the hand is lowered");
// status reports the leader
v.reset(); feed("F", 0.9, 3); assert.strictEqual(v.status().label, "F"); assert.strictEqual(v.status().count, 3);

// Holistic layout still correct
const pts = (n, b, vis) => Array.from({ length: n }, (_, i) => vis ? { x: b + i, y: b, z: b, visibility: 0.9 } : { x: b + i, y: b, z: b });
const f = hf.holisticFrameVector({ poseLandmarks: pts(33, 1, true), faceLandmarks: pts(468, 2), leftHandLandmarks: pts(21, 3), rightHandLandmarks: pts(21, 4) });
assert.strictEqual(f.length, 1662); assert.strictEqual(f[1599], 4);
console.log("LIVE LOGIC TESTS PASSED");
