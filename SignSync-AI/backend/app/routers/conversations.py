from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import get_db
from ..deps import get_current_user
from ..models import Conversation, Message, User
from ..schemas import ConversationCreate, ConversationOut, MessageCreate, MessageOut
from ..services.conversation import get_owned_conversation

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=ConversationOut, status_code=201)
def create_conversation(
    payload: ConversationCreate, user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    conv = Conversation(user_id=user.id, title=payload.title)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


@router.get("", response_model=list[ConversationOut])
def list_conversations(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return db.scalars(
        select(Conversation).where(Conversation.user_id == user.id).order_by(Conversation.id.desc())
    ).all()


@router.get("/{conversation_id}/messages", response_model=list[MessageOut])
def list_messages(
    conversation_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    conv = get_owned_conversation(db, user, conversation_id)
    return conv.messages


@router.post("/{conversation_id}/messages", response_model=MessageOut, status_code=201)
def add_message(
    conversation_id: int,
    payload: MessageCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conv = get_owned_conversation(db, user, conversation_id)
    msg = Message(conversation_id=conv.id, direction=payload.direction, text=payload.text, meta=payload.meta)
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


@router.delete("/{conversation_id}", status_code=204)
def delete_conversation(
    conversation_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    conv = get_owned_conversation(db, user, conversation_id)
    db.delete(conv)
    db.commit()
