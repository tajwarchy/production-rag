"""
Query route.

POST /api/v1/query — retrieve + generate answer for a user question.

The handler is intentionally thin:
  - Validate input
  - Resolve user collection (multi-user isolation via dep injection)
  - Delegate to retrieval_service.retrieve_and_answer()
  - Log the retrieval for RAGAS evaluation
  - Return structured response
"""

import time

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.api.deps import get_current_user_id, get_current_user_collection
from app.db.sqlite import log_retrieval
from app.services.retrieval_service import retrieve_and_answer

router = APIRouter()

VALID_STRATEGIES = {"similarity", "mmr", "hyde"}


# ------------------------------------------------------------------ #
#  Request / response schemas                                          #
# ------------------------------------------------------------------ #

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    strategy: str = Field(
        default="similarity",
        description="Retrieval strategy: similarity | mmr | hyde",
    )


class QueryResponse(BaseModel):
    answer: str
    rewritten_query: str
    strategy: str
    chunks_used: list[str]
    latency_ms: int
    fallback: bool


# ------------------------------------------------------------------ #
#  Route                                                               #
# ------------------------------------------------------------------ #

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question against your ingested documents",
)
async def query_route(
    body: QueryRequest,
    user_id: str = Depends(get_current_user_id),
    collection: str = Depends(get_current_user_collection),
):
    if body.strategy not in VALID_STRATEGIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid strategy '{body.strategy}'. Choose from: {VALID_STRATEGIES}",
        )

    start = time.perf_counter()

    result = retrieve_and_answer(
        question=body.question,
        collection=collection,
        strategy=body.strategy,
    )

    latency_ms = int((time.perf_counter() - start) * 1000)

    # Log to SQLite for RAGAS evaluation dataset
    await log_retrieval(
        user_id=user_id,
        query_original=body.question,
        strategy=body.strategy,
        query_rewritten=result["rewritten_query"],
        num_chunks=len(result["chunks_used"]),
        latency_ms=latency_ms,
    )

    return QueryResponse(
        answer=result["answer"],
        rewritten_query=result["rewritten_query"],
        strategy=result["strategy"],
        chunks_used=result["chunks_used"],
        latency_ms=latency_ms,
        fallback=result["fallback"],
    )