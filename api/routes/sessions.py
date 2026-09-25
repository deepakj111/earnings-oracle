"""
Session management routes — persistent conversational chat history for multi-user SaaS.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, status

from api.chat_store import get_chat_store
from api.models import (
    SessionCreate,
    SessionDetailOut,
    SessionSummaryOut,
    SessionUpdate,
)

router = APIRouter()


def _get_user_id(request: Request) -> str:
    """Extract authenticated or guest user ID from request state or headers."""
    return (
        getattr(request.state, "user_id", None)
        or request.headers.get("x-user-id", "").strip()
        or "default_user"
    )


@router.get(
    "",
    response_model=list[SessionSummaryOut],
    summary="List chat sessions",
    description="Retrieve all active chat sessions for the current authenticated or guest user.",
)
async def list_sessions(
    request: Request,
    limit: int = Query(default=50, ge=1, le=100, description="Max sessions to return"),
) -> list[SessionSummaryOut]:
    user_id = _get_user_id(request)
    store = get_chat_store()
    return store.list_sessions(user_id=user_id, limit=limit)


@router.post(
    "",
    response_model=SessionDetailOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new chat session",
    description="Initialize a new conversational session for multi-turn financial analysis.",
)
async def create_session(
    body: SessionCreate,
    request: Request,
) -> SessionDetailOut:
    user_id = _get_user_id(request)
    tenant_id = getattr(request.state, "tenant_id", "default_tenant")
    store = get_chat_store()
    sid = store.create_session(
        user_id=user_id,
        title=body.title,
        metadata=body.metadata,
        tenant_id=tenant_id,
    )
    session = store.get_session(sid, user_id=user_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve created session",
        )
    return session


@router.get(
    "/{session_id}",
    response_model=SessionDetailOut,
    summary="Get session details and messages",
    description="Retrieve full conversational turns and citations for a given session ID.",
)
async def get_session(
    session_id: str,
    request: Request,
) -> SessionDetailOut:
    user_id = _get_user_id(request)
    store = get_chat_store()
    session = store.get_session(session_id, user_id=user_id)
    if not session:
        # Fall back without user check if guest / shared mode
        session = store.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat session '{session_id}' not found",
            )
    return session


@router.patch(
    "/{session_id}",
    summary="Rename a chat session",
    description="Update the display title of an active chat session.",
)
async def rename_session(
    session_id: str,
    body: SessionUpdate,
    request: Request,
) -> dict[str, str]:
    user_id = _get_user_id(request)
    store = get_chat_store()
    success = store.update_session_title(session_id=session_id, title=body.title, user_id=user_id)
    if not success:
        # Retry without user scoping in case of guest / multi-tab access
        success = store.update_session_title(session_id=session_id, title=body.title)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session '{session_id}' not found or update failed",
        )
    return {"status": "ok", "session_id": session_id, "title": body.title}


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a chat session",
    description="Permanently delete a chat session and all its stored conversational messages.",
)
async def delete_session(
    session_id: str,
    request: Request,
) -> None:
    user_id = _get_user_id(request)
    store = get_chat_store()
    deleted = store.delete_session(session_id, user_id=user_id)
    if not deleted:
        deleted = store.delete_session(session_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session '{session_id}' not found",
        )


@router.post(
    "/{session_id}/clear",
    summary="Clear messages from session",
    description="Clear all conversation messages while preserving the session container.",
)
async def clear_session(
    session_id: str,
    request: Request,
) -> dict[str, str]:
    user_id = _get_user_id(request)
    store = get_chat_store()
    cleared = store.clear_session(session_id, user_id=user_id)
    if not cleared:
        cleared = store.clear_session(session_id)
    if not cleared:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session '{session_id}' not found",
        )
    return {"status": "ok", "session_id": session_id}
