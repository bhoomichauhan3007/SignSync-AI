"""
SignSync AI - CV pipeline tests (no camera, no MediaPipe models needed).

Run from computer_vision/:   python test_cv_pipeline.py     (or: pytest test_cv_pipeline.py)

They use synthetic landmark data, so they prove the schema / preprocessing /
export / live-window LOGIC. They do NOT prove MediaPipe detection quality or
model accuracy.
"""

import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import feature_schema as fs
import sequence_utils as su


def fake_sequence(frames=40, left=True, right=True, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((frames, fs.FEATURES_PER_FRAME), dtype=np.float32)
    pose = rng.uniform(0.2, 0.8, (frames, 33, 4)).astype(np.float32)
    pose[:, :, 3] = 0.95
    X[:, 0:132] = pose.reshape(frames, -1)
    X[:, 132:1536] = rng.uniform(0.3, 0.7, (frames, 1404))
    if left:
        X[:, 1536:1599] = rng.uniform(0.5, 0.9, (frames, 63))
    if right:
        X[:, 1599:1662] = rng.uniform(0.1, 0.5, (frames, 63))
    return X


def test_schema_constants():
    assert fs.FEATURES_PER_FRAME == 1662
    assert fs.POSE_FEATURES + fs.FACE_FEATURES + 2 * fs.HAND_FEATURES == 1662
    ends = [v[1] for v in fs.SECTIONS.values()]
    assert ends == sorted(ends) and ends[-1] == 1662


def test_mask_marks_missing_and_low_visibility():
    X = fake_sequence(10, left=False)
    X[0, 3] = 0.1                       # pose landmark 0 not visible
    m = fs.build_landmark_mask(X)
    assert m.shape == X.shape
    assert m[:, 1536:1599].sum() == 0   # left hand missing everywhere
    assert m[:, 1599:1662].all()        # right hand present
    assert m[0, 0:4].sum() == 0 and m[1, 0:4].all()
    assert (X == fake_sequence(10, left=False)).sum() > 0  # features untouched


def test_fit_to_length():
    f, m, n = su.fit_to_length(fake_sequence(100))
    assert f.shape == (30, 1662) and n == 30 and m.all() is not None
    f, m, n = su.fit_to_length(fake_sequence(12))
    assert n == 12 and not f[12:].any() and not m[12:].any()
    # sub-sampled frames must be real frames, not blends
    X = fake_sequence(100)
    f, _, _ = su.fit_to_length(X)
    assert all(any(np.array_equal(row, x) for x in X) for row in f)


def test_validate_sequence():
    assert su.validate_sequence(fake_sequence(40)) == []
    assert su.validate_sequence(fake_sequence(40, left=False, right=False))
    assert su.validate_sequence(np.zeros((40, 100)))
    bad = fake_sequence(40); bad[3, 5] = np.nan
    assert su.validate_sequence(bad)


def test_normalization_keeps_missing_zero():
    X = fake_sequence(10, left=False)
    n, ok = su.normalize_landmarks(X)
    assert ok.all() and not n[:, 1536:1599].any()
    assert np.allclose(n[:, 0:132].reshape(10, 33, 4)[:, :, 3], 0.95)  # visibility untouched


def test_extract_frame_features_order():
    # stub mediapipe so holistic_features can be imported without the real models
    mp = types.ModuleType("mediapipe")
    sol = types.SimpleNamespace(
        holistic=types.SimpleNamespace(), drawing_utils=types.SimpleNamespace()
    )
    mp.solutions = sol
    sys.modules.setdefault("mediapipe", mp)
    if not hasattr(sys.modules["mediapipe"], "solutions"):
        sys.modules["mediapipe"].solutions = sol
    import holistic_features as hf

    def lm(n, base, vis=False):
        pts = [types.SimpleNamespace(x=base + i * 1e-3, y=base, z=base,
                                     visibility=0.9) for i in range(n)]
        return types.SimpleNamespace(landmark=pts)

    res = types.SimpleNamespace(
        pose_landmarks=lm(33, 1.0), face_landmarks=lm(468, 2.0),
        left_hand_landmarks=lm(21, 3.0), right_hand_landmarks=None)
    v = hf.extract_frame_features(res)
    assert v.shape == (1662,) and v.dtype == np.float32
    assert abs(v[0] - 1.0) < 1e-6 and abs(v[3] - 0.9) < 1e-6        # pose first, 4 values
    assert abs(v[132] - 2.0) < 1e-6                                  # then face
    assert abs(v[1536] - 3.0) < 1e-6                                 # then left hand
    assert not v[1599:].any()                                        # right hand missing -> zeros


def test_end_to_end_preprocess_export_verify():
    import export_features as ef
    import preprocessing as pp

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        samples, processed, exports = tmp / "samples", tmp / "processed", tmp / "exports"
        for d in (samples, processed, exports):
            d.mkdir()
        pp.SAMPLES_DIR, pp.PROCESSED_DIR = samples, processed
        ef.PROCESSED_DIR, ef.EXPORT_DIR = processed, exports

        cols = ["frame", "timestamp"] + [f"feature_{i}" for i in range(1662)]
        specs = [("hello", 60), ("hello", 20), ("thanks", 45), ("test", 40)]
        for k, (label, frames) in enumerate(specs):
            X = fake_sequence(frames, seed=k)
            df = pd.DataFrame(X, columns=cols[2:])
            df.insert(0, "timestamp", np.arange(frames) / 15)
            df.insert(0, "frame", np.arange(frames))
            df.to_csv(samples / f"s{k}.csv", index=False)
            (samples / f"s{k}.json").write_text(json.dumps({
                "sample_id": f"id{k}", "signer_id": "t1",
                "recording_type": "word", "label": label,
                "schema_name": fs.SCHEMA_NAME}))

        # a recording with the OLD 258 schema must be rejected, not silently used
        old = pd.DataFrame(np.random.rand(20, 258))
        old.insert(0, "timestamp", 0); old.insert(0, "frame", 0)
        old.to_csv(samples / "old.csv", index=False)
        (samples / "old.json").write_text(json.dumps({
            "sample_id": "o", "signer_id": "t1", "recording_type": "word", "label": "hello"}))

        pp.preprocess_all_samples()
        assert len(list(processed.glob("*_processed.npz"))) == 3   # 'test' label + 258-schema rejected

        ef.export_recorded_sequences()
        with np.load(exports / "cv_word_dataset.npz") as d:
            assert d["X"].shape == (3, 30, 1662) and d["M"].shape == d["X"].shape
            assert list(d["classes"]) == ["hello", "thanks"]
            assert sorted(d["lengths"].tolist()) == [20, 30, 30]

        import verify_schema_compat as vc
        sys.argv = ["x", "--ml", str(exports / "cv_word_dataset.npz"),
                    "--cv", str(exports / "cv_word_dataset.npz")]
        try:
            vc.main()
        except SystemExit as e:
            assert e.code == 0


def test_live_predictor_logic():
    from live_predictor import LivePredictor

    classes = ["hello", "thanks"]
    calls = []

    def predict(seq):
        calls.append(seq.shape)
        return np.array([0.9, 0.1])

    p = LivePredictor(predict, classes, stable_votes=3, predict_every=5)
    words = [p.push(f) for f in fake_sequence(80)]
    assert calls and all(c == (30, 1662) for c in calls)
    assert [w for w in words if w] == ["hello"]            # emitted once, after 3 agreeing votes

    q = LivePredictor(predict, classes)                    # hands out of view -> no prediction
    n = len(calls)
    for f in fake_sequence(80, left=False, right=False):
        q.push(f)
    assert len(calls) == n

    low = LivePredictor(lambda s: np.array([0.55, 0.45]), classes)
    assert not any(low.push(f) for f in fake_sequence(80))  # below confidence threshold

    wrong = LivePredictor(lambda s: np.ones(5) / 5, classes)
    try:
        for f in fake_sequence(40):
            wrong.push(f)
        raise AssertionError("class-count mismatch not detected")
    except ValueError:
        pass


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print("PASS", name)
            except Exception as exc:
                failed += 1; print("FAIL", name, "->", type(exc).__name__, exc)
    sys.exit(1 if failed else 0)
