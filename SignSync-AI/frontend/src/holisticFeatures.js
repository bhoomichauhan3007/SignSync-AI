// Converts MediaPipe Holistic results into the feature layout Member 1 defines in
// computer_vision/holistic_features.py (and Member 2's word models were trained on):
//   pose 33 x (x, y, z, visibility) = 132
//   face 468 x (x, y, z)            = 1404
//   left hand 21 x (x, y, z)        = 63
//   right hand 21 x (x, y, z)       = 63
//   total 1662 values per frame, in that order. A missing part is filled with zeros,
//   as in Member 1's code.

export const POSE_POINTS = 33;
export const FACE_POINTS = 468;
export const HAND_POINTS = 21;
export const FRAME_FEATURES = 1662;
export const HAND_FEATURES = 63;

function pointsToValues(points, perPoint) {
  const out = new Array(points.length * perPoint);
  for (let i = 0; i < points.length; i += 1) {
    const p = points[i];
    out[i * perPoint] = p.x;
    out[i * perPoint + 1] = p.y;
    out[i * perPoint + 2] = p.z ?? 0;
    if (perPoint === 4) out[i * perPoint + 3] = p.visibility ?? 0;
  }
  return out;
}

function part(points, expectedCount, perPoint) {
  const size = expectedCount * perPoint;
  if (!points || points.length !== expectedCount) return new Array(size).fill(0);
  return pointsToValues(points, perPoint);
}

// One full frame: 1662 values, same order as Member 1's holistic_features.py.
export function holisticFrameVector(results) {
  const values = [
    ...part(results?.poseLandmarks, POSE_POINTS, 4),
    ...part(results?.faceLandmarks, FACE_POINTS, 3),
    ...part(results?.leftHandLandmarks, HAND_POINTS, 3),
    ...part(results?.rightHandLandmarks, HAND_POINTS, 3),
  ];
  return values.map(v => (Number.isFinite(v) ? v : 0));
}

// The hand used for single-letter recognition: the right hand if visible, otherwise the left.
// Member 1's alphabet features are 21 hand points (x, y, z), which is what this returns.
export function handFeatureVector(results) {
  const hand = results?.rightHandLandmarks?.length === HAND_POINTS
    ? results.rightHandLandmarks
    : results?.leftHandLandmarks?.length === HAND_POINTS
      ? results.leftHandLandmarks
      : null;
  if (!hand) return null;
  return pointsToValues(hand, 3).map(v => (Number.isFinite(v) ? v : 0));
}

export function hasAnyHand(results) {
  return Boolean(results?.leftHandLandmarks || results?.rightHandLandmarks);
}
