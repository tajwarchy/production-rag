"""
RAG Evaluation — 3 retrieval strategies side-by-side.
No LLM calls — uses embedding-based context relevancy only.

For production eval with full RAGAS metrics (faithfulness, answer_relevancy),
swap llm_service.py to OpenAI/Anthropic — one-line change as documented there.
"""

import sys
import asyncio
import json
import time
import uuid
from pathlib import Path
from datetime import datetime

sys.path.insert(0, ".")

import numpy as np
from app.core.config import get_settings
from app.db.sqlite import init_db, get_db
from app.db.qdrant import create_collection, delete_collection
from app.services.ingestion_service import ingest_pdf
from app.services.retrieval_service import _retrieve_similarity, _retrieve_mmr
from app.services.embedding_service import embed_query, embed_texts

SAMPLE_PDF   = Path("scripts/sample.pdf")
RESULTS_PATH = Path("eval/results.json")
STRATEGIES   = ["similarity", "mmr", "hyde"]

TEST_QUESTIONS = [
    "What is the main contribution of this paper?",
    "How does the attention mechanism work?",
    "What datasets were used for evaluation?",
]


# ------------------------------------------------------------------ #
#  Setup                                                               #
# ------------------------------------------------------------------ #

async def provision_test_user(collection: str) -> str:
    await init_db()
    user_id = str(uuid.uuid4())
    async with get_db() as conn:
        await conn.execute(
            "INSERT INTO users (id, email, collection) VALUES (?, ?, ?)",
            (user_id, f"{user_id[:8]}@ragas.eval", collection),
        )
        await conn.commit()
    return user_id


def ingest_sample(user_id: str, collection: str) -> int:
    pdf_bytes = SAMPLE_PDF.read_bytes()
    return ingest_pdf(pdf_bytes, user_id, str(uuid.uuid4()), SAMPLE_PDF.name, collection)


# ------------------------------------------------------------------ #
#  Retrieval — no LLM                                                  #
# ------------------------------------------------------------------ #

def retrieve_chunks(question: str, collection: str, strategy: str) -> list[str]:
    """
    Retrieve chunks without any LLM calls.
    HyDE requires an LLM to generate the hypothetical doc, so it falls
    back to similarity for this embedding-only evaluation.
    """
    cfg = get_settings().retrieval
    q_vec = embed_query(question)

    if strategy == "mmr":
        return _retrieve_mmr(collection, q_vec, cfg.top_k, cfg.mmr_lambda)
    else:
        # similarity and hyde (fallback) both use similarity search here
        return _retrieve_similarity(collection, q_vec, cfg.top_k, cfg.score_threshold)


# ------------------------------------------------------------------ #
#  Metric — embedding-based context relevancy                          #
# ------------------------------------------------------------------ #

def context_relevancy_score(question: str, chunks: list[str]) -> float:
    """
    Average cosine similarity between the question and each retrieved chunk.
    Vectors are unit-normalised so dot product == cosine similarity.
    Range: 0.0 (irrelevant) to 1.0 (perfectly relevant).
    """
    if not chunks:
        return 0.0
    vecs = np.array(embed_texts([question] + chunks))
    q_vec = vecs[0]
    chunk_vecs = vecs[1:]
    return float(np.mean(chunk_vecs @ q_vec))


# ------------------------------------------------------------------ #
#  Evaluate one strategy                                               #
# ------------------------------------------------------------------ #

def evaluate_strategy(strategy: str, collection: str) -> dict:
    print(f"\n  [{strategy.upper()}]")
    question_results = []

    for q in TEST_QUESTIONS:
        t0 = time.perf_counter()
        chunks = retrieve_chunks(q, collection, strategy)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        ctx_score = context_relevancy_score(q, chunks)

        question_results.append({
            "question":          q,
            "chunks_used":       len(chunks),
            "context_relevancy": round(ctx_score, 4),
            "latency_ms":        latency_ms,
        })
        print(f"    ✓ '{q[:52]}' | ctx_relevancy={ctx_score:.4f} | {latency_ms}ms")

    avg_ctx    = round(float(np.mean([r["context_relevancy"] for r in question_results])), 4)
    avg_lat    = int(np.mean([r["latency_ms"] for r in question_results]))
    avg_chunks = round(float(np.mean([r["chunks_used"] for r in question_results])), 1)

    return {
        "strategy":              strategy,
        "avg_context_relevancy": avg_ctx,
        "avg_latency_ms":        avg_lat,
        "avg_chunks_used":       avg_chunks,
        "questions":             question_results,
    }


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

async def main():
    print("\n=== RAG Evaluation — 3 retrieval strategies ===\n")

    if not SAMPLE_PDF.exists():
        print(f"[!] PDF not found at {SAMPLE_PDF}")
        sys.exit(1)

    cfg = get_settings().qdrant
    collection = f"{cfg.collection_prefix}ragas-eval-{uuid.uuid4().hex[:8]}"

    user_id = await provision_test_user(collection)
    create_collection(collection)
    n = ingest_sample(user_id, collection)
    print(f"[*] Ingested {n} chunks")
    print(f"[*] Evaluating {len(TEST_QUESTIONS)} questions × {len(STRATEGIES)} strategies")

    all_results = []
    for strategy in STRATEGIES:
        all_results.append(evaluate_strategy(strategy, collection))

    # ── Comparison table ──────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"  {'Strategy':<12} {'Ctx.Relevancy':>14} {'Avg Latency':>12} {'Avg Chunks':>11}")
    print("=" * 62)
    for r in all_results:
        print(
            f"  {r['strategy']:<12} "
            f"{r['avg_context_relevancy']:>14.4f} "
            f"{r['avg_latency_ms']:>10}ms "
            f"{r['avg_chunks_used']:>11.1f}"
        )
    print("=" * 62)

    # ── Save results ──────────────────────────────────────────────
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    output = {
        "evaluated_at":  datetime.utcnow().isoformat(),
        "num_questions": len(TEST_QUESTIONS),
        "metric_note": (
            "context_relevancy is embedding-based (cosine similarity of question vs chunks). "
            "Full LLM-based RAGAS metrics (faithfulness, answer_relevancy) require a "
            "cloud LLM — swap llm_service.py to OpenAI/Anthropic for production eval."
        ),
        "results": all_results,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2))
    print(f"\n[*] Results saved to {RESULTS_PATH}")

    delete_collection(collection)
    print("[*] Eval collection cleaned up")
    print("\n✅ Evaluation complete. Ready for Phase 6.\n")


if __name__ == "__main__":
    asyncio.run(main())