"""
Simulates Qdrant being down and verifies the graceful fallback response.

How it works:
  Monkeypatches is_qdrant_healthy() to return False, then calls
  retrieve_and_answer() — which should return the fallback message
  instead of raising an exception or hanging.

Run: python scripts/verify_fallback.py
No Docker changes needed — this is pure code-level simulation.
"""

import sys
sys.path.insert(0, ".")

import app.services.retrieval_service as retrieval_module
from app.services.retrieval_service import retrieve_and_answer, FALLBACK_RESPONSE

# ── Monkeypatch: simulate Qdrant being unreachable ─────────────────
retrieval_module.is_qdrant_healthy = lambda: False


def main():
    print("\n=== Graceful fallback simulation ===\n")
    print("[*] is_qdrant_healthy() patched to return False")
    print("[*] Calling retrieve_and_answer()...\n")

    result = retrieve_and_answer(
        question="What is this document about?",
        collection="user_does_not_matter",
        strategy="similarity",
    )

    print(f"  fallback:        {result['fallback']}")
    print(f"  answer:          {result['answer']}")
    print(f"  chunks_used:     {result['chunks_used']}")

    assert result["fallback"] is True, "Expected fallback=True"
    assert result["answer"] == FALLBACK_RESPONSE, "Expected fallback message"
    assert result["chunks_used"] == [], "Expected empty chunks on fallback"

    print("\n✅ Graceful fallback working correctly.")
    print("   Qdrant down → safe response returned, no crash, no hang.\n")

    # ── Real-world equivalent ──────────────────────────────────────
    print("Real-world equivalent:")
    print("  docker-compose stop qdrant")
    print("  curl -X POST http://localhost:8000/api/v1/query \\")
    print('    -H "X-User-Id: <your-user-id>" \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"question": "test", "strategy": "similarity"}\'')
    print("  → returns fallback JSON instead of 500 error")
    print("  docker-compose start qdrant   # bring it back\n")


if __name__ == "__main__":
    main()