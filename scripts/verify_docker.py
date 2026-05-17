"""
Phase 6 — Docker health check verification.
Confirms all 3 services are reachable before running the full stack.

Run: python scripts/verify_docker.py
"""

import sys
import subprocess
sys.path.insert(0, ".")

import httpx
from app.db.qdrant import is_qdrant_healthy
import redis as redis_lib


def check(label: str, ok: bool, detail: str = ""):
    status = "✓" if ok else "✗"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return ok


def main():
    print("\n=== Docker service health check ===\n")
    all_ok = True

    # 1. Qdrant
    qdrant_ok = is_qdrant_healthy()
    all_ok &= check("Qdrant (localhost:6333)", qdrant_ok,
                    "reachable" if qdrant_ok else "UNREACHABLE — run: docker-compose up -d")

    # 2. Redis
    try:
        r = redis_lib.Redis(host="localhost", port=6379)
        r.ping()
        redis_ok = True
    except Exception as e:
        redis_ok = False
    all_ok &= check("Redis  (localhost:6379)", redis_ok,
                    "reachable" if redis_ok else "UNREACHABLE — run: docker-compose up -d")

    # 3. FastAPI
    try:
        resp = httpx.get("http://localhost:8000/health", timeout=5)
        api_ok = resp.status_code == 200
        detail = resp.json().get("version", "") if api_ok else f"HTTP {resp.status_code}"
    except Exception:
        api_ok = False
        detail = "UNREACHABLE — run: uvicorn app.main:app --reload --port 8000"
    all_ok &= check("FastAPI (localhost:8000)", api_ok, detail)

    # 4. Qdrant dashboard
    try:
        resp = httpx.get("http://localhost:6333/dashboard", timeout=5)
        dash_ok = resp.status_code == 200
    except Exception:
        dash_ok = False
    all_ok &= check("Qdrant dashboard", dash_ok, "http://localhost:6333/dashboard")

    print()
    if all_ok:
        print("✅ All services healthy. Stack is ready.\n")
    else:
        print("❌ One or more services are down. Fix above, then rerun.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()