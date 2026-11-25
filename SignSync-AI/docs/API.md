# SignSync AI - API for the AI Tutor (Member 4)

Base URL (local): `http://localhost:8000`. Interactive docs: `/docs`.

## Authentication
All assistant routes need a Bearer token from `POST /auth/login`.
`Authorization: Bearer <access_token>`

## POST /assistant/ask

Ask the ISL tutor a question. If `conversation_id` is given, earlier questions and answers in that
conversation are passed to the model as history and the exchange is saved.

Request (`AssistantRequest`):
| Field | Type | Required | Rules |
|---|---|---|---|
| question | string | yes | 1 to 1000 characters |
| conversation_id | integer or null | no | must belong to the logged-in user, otherwise 404 |

Response (`AssistantResponse`):
| Field | Type | Meaning |
|---|---|---|
| answer | string | the tutor's reply, or a refusal, or a fallback with the reason |
| sources | array of objects | passages used. Each has `title`, `page` (int or null), `score` (0 to 1, rounded to 3 decimals), `status` (`verified`, `sample` or `unverified`), `source` (file path inside the knowledge base) |
| source | string | `real` when the RAG module answered, `mock` when the backend is in mock mode |

Example:

    curl -X POST http://localhost:8000/assistant/ask \
      -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
      -d '{"question": "What should I do in an emergency in India?"}'

Response (extractive mode, sample knowledge base):

    {
      "answer": "If this is a real emergency, contact local emergency services immediately. In India, the emergency number is 112.\n\nMost relevant passages ...",
      "sources": [{"title": "SAMPLE TEST DOCUMENT ...", "page": null, "score": 0.098, "status": "sample", "source": "sample/sample_project_info.md"}],
      "source": "real"
    }

Behaviour guarantees:
- Off-topic or unsupported questions return a refusal with `sources: []`.
- Questions about "this sign" get a note that the tutor cannot see video.
- If the LLM is unavailable, the answer contains the reason and the matching passages.
- If the index has not been built, `answer` says how to build it and `sources` is empty. The HTTP status stays 200.

Errors: 401 (missing or expired token), 404 (unknown `conversation_id`), 422 (invalid body).

## Internal Python contract (used by app/services/ai_adapter.py)

    from rag.assistant import ask
    ask(question: str, history: list[dict] | None = None) -> dict
    # history items: {"role": "user" | "assistant", "content": str}
    # returns: {"answer": str, "sources": list[dict], "mode": "llm" | "extractive" | "refused" | "unavailable"}

`mode` is not forwarded by the API response (the schema keeps `answer`, `sources`, `source`). It is for logs and tests.

## Health check

`GET /health/ai` returns the status of each AI hook. After enabling `AI_MODE=real`, `ask_assistant` should show `real`.
