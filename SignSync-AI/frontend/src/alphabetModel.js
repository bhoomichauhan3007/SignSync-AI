// Runs Member 2's alphabet model in the browser (weights exported by backend/ml/export_browser_model.py).
// Same arithmetic as the Python model: standardise, two ReLU layers, softmax. No server call is needed.

let modelPromise = null;

export function loadAlphabetModel() {
  if (!modelPromise) {
    modelPromise = fetch(`${import.meta.env.BASE_URL}models/alphabet.json`)
      .then(response => {
        if (!response.ok) throw new Error("The letter model file could not be loaded.");
        return response.json();
      });
  }
  return modelPromise;
}

// vector: 63 numbers (21 hand points x, y, z) in the order Member 1 defines.
// Returns the most likely letter, its confidence and the top three candidates.
export function classifyHand(model, vector) {
  let activations = vector.map((value, i) => (value - model.mean[i]) / model.scale[i]);
  const layerCount = model.layers.length;
  model.layers.forEach((layer, index) => {
    const outputs = layer.bias.slice();
    for (let i = 0; i < activations.length; i += 1) {
      const a = activations[i];
      const row = layer.weights[i];
      for (let j = 0; j < outputs.length; j += 1) outputs[j] += a * row[j];
    }
    if (index < layerCount - 1) {
      for (let j = 0; j < outputs.length; j += 1) outputs[j] = Math.max(0, outputs[j]);
    }
    activations = outputs;
  });
  const top = Math.max(...activations);
  const exps = activations.map(v => Math.exp(v - top));
  const total = exps.reduce((sum, v) => sum + v, 0);
  const probabilities = exps.map(v => v / total);
  const ranked = probabilities
    .map((p, i) => ({ label: model.labels[i], confidence: p }))
    .sort((a, b) => b.confidence - a.confidence);
  return { label: ranked[0].label, confidence: ranked[0].confidence, top: ranked.slice(0, 3) };
}

// Flattens MediaPipe hand landmarks (21 points with x, y, z) into the 63-value vector.
export function handToVector(landmarks) {
  const values = [];
  for (const p of landmarks) values.push(p.x, p.y, p.z ?? 0);
  return values;
}
