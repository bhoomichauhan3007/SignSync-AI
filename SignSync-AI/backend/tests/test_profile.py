"""Sign-up saves the learner's profile; the profile endpoint returns it."""


def _payload(**profile_overrides):
    profile = {
        "who": "learner",
        "age_group": "18_25",
        "experience": "never",
        "goal": "school_college",
        "problem": "I need signs for my exam.",
        "consent": True,
    }
    profile.update(profile_overrides)
    return {"name": "Profile Learner", "email": "profile.learner@example.com",
            "password": "longenough1", "profile": profile}


def test_register_with_profile_saves_it(client):
    response = client.post("/auth/register", json=_payload())
    assert response.status_code == 201
    token = response.json()["access_token"]
    profile = client.get("/users/me/profile", headers={"Authorization": f"Bearer {token}"})
    assert profile.status_code == 200
    body = profile.json()
    assert body["goal"] == "school_college"
    assert body["problem"] == "I need signs for my exam."


def test_profile_requires_consent(client):
    response = client.post("/auth/register", json=_payload(consent=False, goal="school_college"))
    assert response.status_code == 422


def test_profile_rejects_unknown_option(client):
    response = client.post("/auth/register", json=_payload(who="alien"))
    assert response.status_code == 422
