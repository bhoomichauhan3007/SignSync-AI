from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database import get_db
from ..deps import get_current_user
from ..models import User
from ..schemas import AssistantRequest, AssistantResponse
from ..services import ai_adapter as ai
from ..services.conversation import get_owned_conversation, log_message

router = APIRouter(prefix="/assistant", tags=["assistant"])


@router.post("/ask", response_model=AssistantResponse)
def ask(payload: AssistantRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """RAG-based ISL tutor. If a conversation_id is given, earlier Q&A is passed as history."""
    history = []
    if payload.conversation_id is not None:
        conv = get_owned_conversation(db, user, payload.conversation_id)
        history = [
            {"role": "user" if m.direction == "assistant_q" else "assistant", "content": m.text}
            for m in conv.messages
            if m.direction in ("assistant_q", "assistant_a")
        ][-10:]

    out = ai.ask_assistant(payload.question, history)
    log_message(db, user, payload.conversation_id, "assistant_q", payload.question)
    log_message(db, user, payload.conversation_id, "assistant_a", out["answer"], {"sources": out["sources"]})
    db.commit()
    return out
