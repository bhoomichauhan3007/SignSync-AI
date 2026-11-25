import os
import uuid

# Must be set BEFORE the app is imported (settings are read at import time)
if os.path.exists("test_signsync.db"):
    os.remove("test_signsync.db")
os.environ["DATABASE_URL"] = "sqlite:///./test_signsync.db"
os.environ["SECRET_KEY"] = "test-secret-key-for-pytest-only-32bytes!"
os.environ["AI_MODE"] = "mock"
os.environ["LOGIN_RATE_LIMIT"] = "1000"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.database import engine  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:  # runs lifespan: creates tables + seeds
        yield c
    engine.dispose()
    if os.path.exists("test_signsync.db"):
        os.remove("test_signsync.db")


@pytest.fixture()
def auth(client):
    """Registers a fresh user and returns (headers, token)."""
    email = f"user_{uuid.uuid4().hex[:8]}@example.com"
    r = client.post("/auth/register", json={"name": "Test User", "email": email, "password": "password123"})
    assert r.status_code == 201, r.text
    token = r.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}, token
