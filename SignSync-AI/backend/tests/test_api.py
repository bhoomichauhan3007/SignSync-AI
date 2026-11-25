LANDMARKS = [[0.1, 0.2, 0.3]] * 21


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}
    body = client.get("/health/ai").json()
    assert body["ai_mode"] == "mock"
    assert set(body["hooks"].values()) == {"mock"}


def test_register_login_me(client):
    email = "login_test@example.com"
    r = client.post("/auth/register", json={"name": "Login Test", "email": email, "password": "password123"})
    assert r.status_code == 201
    r = client.post("/auth/login", json={"email": email, "password": "password123"})
    assert r.status_code == 200
    token = r.json()["access_token"]
    me = client.get("/users/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == email
    bad = client.post("/auth/login", json={"email": email, "password": "wrong-password"})
    assert bad.status_code == 401


def test_duplicate_register(client):
    payload = {"name": "Dup", "email": "dup@example.com", "password": "password123"}
    assert client.post("/auth/register", json=payload).status_code == 201
    assert client.post("/auth/register", json=payload).status_code == 409


def test_refresh_token(client):
    r = client.post("/auth/register", json={"name": "Ref", "email": "ref@example.com", "password": "password123"})
    refresh = r.json()["refresh_token"]
    r2 = client.post("/auth/refresh", json={"refresh_token": refresh})
    assert r2.status_code == 200
    # an access token must not work as a refresh token
    r3 = client.post("/auth/refresh", json={"refresh_token": r.json()["access_token"]})
    assert r3.status_code == 401


def test_protected_requires_auth(client):
    assert client.get("/progress/me").status_code == 401


def test_lessons_and_signs_seeded(client):
    lessons = client.get("/lessons").json()
    assert len(lessons) >= 6
    detail = client.get(f"/lessons/{lessons[0]['id']}").json()
    assert len(detail["signs"]) == lessons[0]["sign_count"] > 0
    assert any(s["label"] == "HELLO" for s in client.get("/signs?q=hello").json())


def test_admin_only_sign_creation(client, auth):
    headers, _ = auth
    r = client.post("/signs", json={"label": "TEST", "category": "word"}, headers=headers)
    assert r.status_code == 403
    from app.config import settings  # use the configured admin account, not a hardcoded one
    login = client.post("/auth/login", json={"email": settings.admin_email, "password": settings.admin_password})
    admin = {"Authorization": f"Bearer {login.json()['access_token']}"}
    r = client.post("/signs", json={"label": "test sign", "category": "word"}, headers=admin)
    assert r.status_code == 201
    assert r.json()["label"] == "TEST SIGN"


def test_predict_sign(client, auth):
    headers, _ = auth
    r = client.post("/predict/sign", json={"landmarks": LANDMARKS, "mode": "static"}, headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert {"label", "confidence", "low_confidence", "is_emergency", "source"} <= set(body)
    assert 0 <= body["confidence"] <= 1


def test_evaluate_updates_progress(client, auth):
    headers, _ = auth
    sign = client.get("/signs?q=HELLO").json()[0]
    # try a few different inputs; the mock is deterministic per input
    got_correct = False
    for i in range(10):
        r = client.post(
            "/learn/evaluate",
            json={"sign_id": sign["id"], "landmarks": [[i / 10, 0.2, 0.3]] * 21},
            headers=headers,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert 0 <= body["score"] <= 100
        assert set(body["feedback"]) == {"hand_position", "hand_shape", "movement", "orientation"}
        got_correct = got_correct or body["is_correct"]
    assert got_correct, "mock should produce at least one correct attempt in 10 tries"

    prog = client.get("/progress/me", headers=headers).json()
    assert prog["gamification"]["xp"] > 0
    assert "first_sign" in prog["gamification"]["badges"]
    assert any(l["completion_pct"] > 0 for l in prog["lessons"])
    assert client.get("/progress/recommendations", headers=headers).status_code == 200
    assert len(client.get("/learn/attempts", headers=headers).json()) == 10
    board = client.get("/leaderboard", headers=headers).json()
    assert board and board[0]["rank"] == 1


def test_quiz_flow(client, auth):
    headers, _ = auth
    lesson_id = client.get("/lessons").json()[0]["id"]
    quiz = client.get(f"/quiz/{lesson_id}", headers=headers).json()
    assert quiz["questions"]
    answers = [{"sign_id": q["sign_id"], "chosen_label": q["options"][0]} for q in quiz["questions"]]
    r = client.post(f"/quiz/{lesson_id}/submit", json={"answers": answers}, headers=headers)
    assert r.status_code == 200
    assert r.json()["total"] == len(answers)


def test_translation_endpoints(client, auth):
    headers, _ = auth
    r = client.post("/translate/text-to-sign", json={"text": "Thank you, I need help"}, headers=headers)
    assert r.status_code == 200
    labels = [s["label"] for s in r.json()["sequence"]]
    assert "THANK YOU" in labels and "HELP" in labels
    # unknown words are flagged and get fingerspelling
    unknown = [s for s in r.json()["sequence"] if not s["known"]]
    assert all(s["fingerspell"] for s in unknown)

    r = client.post("/translate/signs-to-sentence", json={"words": ["HELLO", "HELP"]}, headers=headers)
    assert r.status_code == 200 and r.json()["sentence"].startswith("Hello")

    r = client.post("/translate/speech-to-sign", data={"text": "water please"}, headers=headers)
    assert r.status_code == 200
    r = client.post("/translate/speech-to-sign", headers=headers)
    assert r.status_code == 400

    r = client.post("/translate/text-to-speech", json={"text": "hello"}, headers=headers)
    assert r.status_code == 200 and r.json()["audio_available"] is False


def test_conversation_logging(client, auth):
    headers, _ = auth
    conv = client.post("/conversations", json={"title": "Demo"}, headers=headers).json()
    client.post(
        "/translate/text-to-sign", json={"text": "hello", "conversation_id": conv["id"]}, headers=headers
    )
    client.post(
        "/assistant/ask", json={"question": "How can I learn ISL?", "conversation_id": conv["id"]}, headers=headers
    )
    msgs = client.get(f"/conversations/{conv['id']}/messages", headers=headers).json()
    assert [m["direction"] for m in msgs] == ["text_to_sign", "assistant_q", "assistant_a"]


def test_conversations_are_private(client, auth):
    headers, _ = auth
    conv = client.post("/conversations", json={}, headers=headers).json()
    other = client.post(
        "/auth/register", json={"name": "Other", "email": "other@example.com", "password": "password123"}
    ).json()["access_token"]
    r = client.get(f"/conversations/{conv['id']}/messages", headers={"Authorization": f"Bearer {other}"})
    assert r.status_code == 404


def test_websocket_flow(client, auth):
    _, token = auth
    with client.websocket_connect(f"/ws/predict?token={token}") as ws:
        ws.send_json({"type": "landmarks", "landmarks": LANDMARKS, "mode": "static"})
        msg = ws.receive_json()
        assert msg["type"] == "prediction"
        ws.send_json({"type": "ping"})
        assert ws.receive_json()["type"] == "pong"
        ws.send_json({"type": "flush"})
        assert ws.receive_json()["type"] == "sentence"
        ws.send_json({"type": "nonsense"})
        assert ws.receive_json()["type"] == "error"


def test_websocket_rejects_bad_token(client):
    import pytest
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/predict?token=not-a-token"):
            pass
