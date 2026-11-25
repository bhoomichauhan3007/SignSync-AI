from typing import Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from ..models import Conversation, Message, User


def get_owned_conversation(db: Session, user: User, conversation_id: int) -> Conversation:
    conv = db.get(Conversation, conversation_id)
    if conv is None or conv.user_id != user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


def log_message(
    db: Session,
    user: User,
    conversation_id: Optional[int],
    direction: str,
    text: str,
    meta: Optional[dict] = None,
) -> None:
    """Append a message to a conversation if one is given (caller commits)."""
    if conversation_id is None:
        return
    conv = get_owned_conversation(db, user, conversation_id)
    db.add(Message(conversation_id=conv.id, direction=direction, text=text, meta=meta))
