"""
Shared FastAPI dependencies.
Injected via Depends() into route handlers.
Keeps route handlers thin — no business logic here.
"""

from fastapi import Header, HTTPException, status
from app.db.sqlite import get_user, get_user_collection


async def get_current_user_id(x_user_id: str = Header(...)) -> str:
    """
    Reads user identity from the X-User-Id request header.

    In production this would be replaced with JWT validation or
    an OAuth2 bearer token. For this project, the header is the
    deliberate simplification — the multi-user isolation logic
    (SQLite lookup → Qdrant collection routing) is identical either way.
    """
    user = await get_user(x_user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{x_user_id}' not found. POST /api/v1/users to register.",
        )
    return x_user_id


async def get_current_user_collection(x_user_id: str = Header(...)) -> str:
    """
    Resolves the user's Qdrant collection name from SQLite.
    This is the multi-user isolation mechanism:
      request header → SQLite lookup → Qdrant collection name
    Each user's documents are physically isolated in their own collection.
    """
    collection = await get_user_collection(x_user_id)
    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No collection found for user '{x_user_id}'.",
        )
    return collection