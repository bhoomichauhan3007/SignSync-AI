// Node test: browser-style classification must match scikit-learn on the same samples.
const fs = require("fs");
const path = require("path");
const ts = require("/opt/npm-tools/node_modules/typescript");
const assert = require("assert");
const src = fs.readFileSync(path.join(__dirname, "../src/alphabetModel.js"), "utf8");
const out = ts.transpileModule(src, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 } }).outputText
  .replace('import.meta.env.BASE_URL', '"/"');
const m = { exports: {} };
new Function("module", "exports", "require", out)(m, m.exports, require);
const { classifyHand, handToVector } = m.exports;
const model = JSON.parse(fs.readFileSync(path.join(__dirname, "../public/models/alphabet.json"), "utf8"));
const fixture = JSON.parse(fs.readFileSync(path.join(__dirname, "fixtures/alphabet_fixture.json"), "utf8"));
let correct = 0, maxDiff = 0, agree = 0;
fixture.X.forEach((vec, i) => {
  const r = classifyHand(model, vec);
  if (r.label === fixture.y[i]) correct += 1;
  if (r.label === fixture.pred[i]) agree += 1;
  maxDiff = Math.max(maxDiff, Math.abs(r.confidence - fixture.prob_max[i]));
});
const n = fixture.X.length;
assert.strictEqual(agree, n, `browser labels differ from Python on ${n - agree} samples`);
assert.ok(maxDiff < 1e-6, `confidence differs by ${maxDiff}`);
assert.strictEqual(handToVector([{ x: 1, y: 2, z: 3 }]).length, 3);
console.log(`browser model: ${agree}/${n} labels match Python, accuracy ${(correct / n * 100).toFixed(1)}%, max confidence difference ${maxDiff.toExponential(2)}`);
