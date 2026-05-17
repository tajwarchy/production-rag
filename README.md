# Production RAG — LangChain + Qdrant + Docker

A production-grade Retrieval-Augmented Generation system built for RAG / LLM engineering roles.

**Stack:** LangChain · Qdrant · Ollama (Mistral 7B) · FastAPI · Celery · Redis · SQLite · Docker  
**Environment:** MacBook Air M1 · MPS inference · conda

---

## Quick start

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate rag-prod

# 2. Start infrastructure
docker-compose up -d

# 3. Start Ollama (separate terminal)
ollama serve
ollama pull mistral

# 4. Start API (separate terminal)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 5. Start worker (separate terminal)
celery -A app.worker.celery_app.celery_app worker --loglevel=info --pool=solo

# 6. Open API docs
open http://localhost:8000/docs

# 7. Open Qdrant UI
open http://localhost:6333/dashboard
```

---

## System architecture

### Read path (query)

```
Client
  │
  │  POST /api/v1/query  {question, strategy}
  ▼
FastAPI  (stateless)
  │
  ├─► SQLite lookup: user_id → Qdrant collection name
  │
  ▼
Query rewriter  (LangChain chain → Ollama)
  │  ambiguous query → retrieval-optimised query
  ▼
Retrieval service  [similarity | MMR | HyDE]
  │
  ▼
Qdrant  (per-user collection, cosine similarity)
  │  top-k chunks returned
  ▼
Cross-encoder reranker  (local, MPS)
  │  top-k → top-k_rerank chunks
  ▼
LLM service  (Ollama / Mistral 7B)
  │  grounded answer generated
  ▼
Client  ← JSON response {answer, rewritten_query, chunks_used, latency_ms}
```

### Write path (ingestion)

```
Client
  │
  │  POST /api/v1/ingest  (PDF upload)
  ▼
FastAPI
  ├─► SQLite: INSERT document row (status: pending)
  ├─► Celery: enqueue ingest_document task
  └─► HTTP 202 Accepted  ← returns immediately, never blocks

        [background — completely decoupled from query path]

Celery worker
  │
  ▼
PyMuPDF  →  text extraction
  │
  ▼
RecursiveCharacterTextSplitter  →  chunks  (size=512, overlap=64)
  │
  ▼
all-MiniLM-L6-v2  →  embeddings  (MPS, num_workers=0)
  │
  ▼
Qdrant upsert  (per-user collection, with payload metadata)
  │
  ▼
SQLite: UPDATE job status → completed | failed
```

---

## System design checkpoint

### Why is ingestion a background worker and not part of the API handler?

PDF ingestion involves three expensive sequential operations: text extraction, chunking, and embedding. For a 50-page document this can take 30–120 seconds. If this ran inside the API handler, the server thread would be blocked for that entire duration — unable to serve any query requests. With `--workers 1`, the entire API would be frozen.

The background worker pattern solves this completely. The API handler does three cheap things (write a DB row, enqueue a task, return 202), taking under 100ms. The worker runs in a separate process. A crash or slowdown during ingestion is fully isolated from query latency. The client polls `/jobs/{job_id}` for completion status.

This is the standard pattern for any operation that is too slow to run synchronously in a web handler.

---

### What is the role of the SQLite metadata layer? What would you replace it at scale?

SQLite serves as the control plane for everything that Qdrant doesn't own:

- **User → collection mapping**: maps each `user_id` to their Qdrant collection name. Every query routes through this lookup.
- **Document registry**: tracks every uploaded file, its size, and how many chunks were produced.
- **Job lifecycle**: `pending → running → completed | failed`. The API and worker coordinate entirely through this table — no shared memory, no direct coupling.
- **Retrieval audit log**: records every query with strategy, latency, and chunk count. Used for evaluation and debugging.

At scale, SQLite would be replaced with **PostgreSQL** for the metadata (concurrent writes, connection pooling, replication). The job table would move to a proper task queue backend such as Celery with a PostgreSQL result backend, or a dedicated system like Temporal. The retrieval log would move to a time-series store or data warehouse for analytics.

SQLite is the right choice here: zero infrastructure overhead, full ACID compliance, and sufficient for single-node throughput.

---

### This system has 3 services in docker-compose. If query latency spikes, how do you isolate which service is the bottleneck?

Work through the call chain systematically:

1. **Check API logs first** (`docker logs rag-api`). The query route logs `latency_ms` for the full request. If latency is high, the slowness is somewhere downstream of FastAPI.

2. **Check the retrieval step**. Add timing around `embed_query()` and `similarity_search()` separately. If embedding is slow, the bottleneck is the embedding model on MPS. If `similarity_search()` is slow, check Qdrant.

3. **Check Qdrant** (`http://localhost:6333/dashboard → telemetry`). Qdrant exposes per-collection search latency. High search time with a small collection means resource contention — likely the container is memory-starved.

4. **Check the LLM**. Ollama logs every request with its duration. A 30-second LLM call is expected for Mistral 7B locally. If it's timing out, Ollama is overloaded or the context window is too large.

5. **Check Redis** (`redis-cli monitor`). If the ingestion worker is hammering Redis during a large ingest, it can starve the broker of connections and delay task dispatch — which indirectly affects perceived API responsiveness.

The key insight: because the services are separated, you can time each boundary independently. A monolith gives you one latency number. Three services give you three.

---

### What does "stateless API server" mean here and why does it matter for scaling?

The FastAPI service holds zero per-request or per-user state in memory. Every request is fully self-contained:

- User identity comes from the `X-User-Id` header
- The Qdrant collection name is looked up from SQLite on every request
- Embeddings and LLM calls are stateless function calls
- No session objects, no in-memory caches that differ between requests

This means you can run any number of API replicas behind a load balancer and any replica can serve any request. There is no concept of "sticky sessions." If a replica crashes, the load balancer routes to another with zero impact on users.

In docker-compose, `workers: 1` is set because M1 is single-node. In production, you would set `workers: 4` or run multiple container replicas behind an nginx or AWS ALB load balancer. The code requires zero changes.

---

### LangChain one-line swap: how to replace Ollama with OpenAI or Anthropic

In `app/services/llm_service.py`, the entire LLM provider is determined by a single line (marked `── SWAP LINE ──`):

```python
# Current (local, free):
llm = OllamaLLM(model="mistral", base_url="http://localhost:11434", ...)

# Swap to OpenAI (one line):
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# Swap to Anthropic (one line):
llm = ChatAnthropic(model="claude-3-5-haiku-latest", temperature=0.1)
```

Every downstream component — the RAG prompt chain, the query rewriter, the HyDE retriever — uses `BaseLanguageModel` from LangChain core and is completely provider-agnostic. None of them import Ollama. None of them need to change.

This is the core value of LangChain's abstraction layer. Ollama is a fully legitimate local stand-in for portfolio purposes: the architecture, the chains, the retrieval logic, and the evaluation harness are all identical to what you would ship with a cloud provider.

---

### CAP theorem applied to Qdrant

Qdrant is a **CP system** — it prioritises Consistency and Partition tolerance over Availability.

During a network partition or node failure, Qdrant will refuse to serve reads and writes rather than risk returning stale or divergent vector data. This is the correct tradeoff for a retrieval system: returning wrong chunks silently is worse than returning an error that the application can handle.

In this project, `is_qdrant_healthy()` in `app/db/qdrant.py` is called at the start of every query. If Qdrant is unreachable, the API returns a safe fallback response immediately rather than hanging or returning a 500. This is the application-level availability layer that compensates for Qdrant's CP behaviour.

Simulate it:
```bash
docker-compose stop qdrant
# query the API — receive graceful fallback response
docker-compose start qdrant
```

---

## Retrieval strategies

| Strategy | How it works | Best for |
|---|---|---|
| **Similarity** | Cosine nearest-neighbour on query embedding | Fast, precise, specific questions |
| **MMR** | Balances relevance vs diversity (λ=0.5) | Broad questions, avoiding repetition |
| **HyDE** | Embeds a hypothetical answer, searches with that | Queries phrased as questions not keywords |

## RAGAS evaluation results

Evaluated on 3 questions × 3 strategies. Metric: embedding-based context relevancy (cosine similarity of question vs retrieved chunks). Full LLM-based RAGAS metrics require a cloud LLM provider — swap `llm_service.py` to OpenAI/Anthropic for production evaluation.

| Strategy | Ctx. Relevancy | Avg Latency | Avg Chunks |
|---|---|---|---|
| similarity | 0.3616 | 50ms | 10.0 |
| mmr | 0.3004 | 62ms | 10.0 |
| hyde | 0.3616 | 19ms | 10.0 |

**MMR trades relevancy for diversity** — lower score is expected and correct. HyDE fell back to similarity for embedding-only evaluation; with Ollama running, it would generate a hypothetical document first.

---

## Project structure

```
rag-prod/
├── app/
│   ├── api/routes/        # query, ingest, users
│   ├── core/              # config loader, logging
│   ├── db/                # qdrant wrapper, sqlite layer, schema
│   ├── services/          # embedding, retrieval, reranking, LLM, query rewriter
│   └── worker/            # celery app + ingest task
├── eval/
│   └── ragas_eval.py      # 3-strategy benchmark
├── scripts/               # verification scripts
├── config.yaml            # all parameters — nothing hardcoded
├── docker-compose.yml     # qdrant + redis + api + worker
├── Dockerfile.api
├── Dockerfile.worker
└── environment.yml
```

---

## LLM abstraction — Ollama as a local stand-in

Ollama running Mistral 7B locally is a legitimate portfolio choice. The LangChain abstraction means the entire system is provider-agnostic. To switch to OpenAI or Anthropic in production: change one line in `llm_service.py` and set an API key environment variable. Every chain, prompt, and retrieval strategy remains identical.