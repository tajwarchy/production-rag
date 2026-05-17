"""
User registration route.
POST /api/v1/users — creates a user row in SQLite + their Qdrant collection.

Multi-user isolation design:
  Every user gets a dedicated Qdrant collection (user_{uuid}).
  The SQLite users table is the authoritative mapping:
    user_id → collection_name
  At query time, the API reads this mapping and routes the vector search
  to the correct collection — documents from user A are never visible to user B.

  At scale, SQLite would be replaced with PostgreSQL, and collection
  creation would be an async job. For this project SQLite is the right
  choice: zero infrastructure, ACID, and sufficient for single-node load.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr

from app.db.sqlite import create_user, get_user
from app.db.qdrant import create_collection

router = APIRouter()


class UserCreateRequest(BaseModel):
    email: EmailStr


class UserResponse(BaseModel):
    id: str
    email: str
    collection: str


@router.post(
    "/users",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user and provision their Qdrant collection",
)
async def create_user_route(body: UserCreateRequest):
    try:
        user = await create_user(body.email)
    except Exception as exc:
        if "UNIQUE constraint failed" in str(exc):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Email '{body.email}' is already registered.",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    # Provision Qdrant collection for this user (idempotent)
    create_collection(user["collection"])

    return UserResponse(**user)


@router.get(
    "/users/{user_id}",
    response_model=UserResponse,
    summary="Get user info",
)
async def get_user_route(user_id: str):
    user = await get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{user_id}' not found.",
        )
    return UserResponse(**user)