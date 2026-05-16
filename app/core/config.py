from functools import lru_cache
from pathlib import Path
import yaml
from pydantic import BaseModel


# ------------------------------------------------------------------ #
#  Sub-models — one per top-level key in config.yaml                  #
# ------------------------------------------------------------------ #

class AppConfig(BaseModel):
    title: str
    version: str
    debug: bool


class QdrantConfig(BaseModel):
    host: str
    port: int
    grpc_port: int
    collection_prefix: str
    vector_size: int
    distance: str
    timeout_seconds: int


class EmbeddingConfig(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str
    device: str
    num_workers: int
    batch_size: int


class ChunkingConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int
    separators: list[str]


class RetrievalConfig(BaseModel):
    strategy: str
    top_k: int
    top_k_rerank: int
    mmr_lambda: float
    score_threshold: float


class RerankerConfig(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str
    device: str
    max_length: int


class LLMConfig(BaseModel):
    provider: str
    model: str
    base_url: str
    temperature: float
    max_tokens: int
    request_timeout: int


class QueryRewriterConfig(BaseModel):
    enabled: bool
    model: str


class DatabaseConfig(BaseModel):
    path: str


class WorkerConfig(BaseModel):
    broker_url: str
    result_backend: str
    task_time_limit: int
    task_soft_time_limit: int


class ServerConfig(BaseModel):
    host: str
    port: int
    workers: int


class RagasConfig(BaseModel):
    test_dataset_path: str
    metrics: list[str]
    output_path: str


# ------------------------------------------------------------------ #
#  Root config — mirrors the top-level structure of config.yaml       #
# ------------------------------------------------------------------ #

class Settings(BaseModel):
    app: AppConfig
    qdrant: QdrantConfig
    embedding: EmbeddingConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    reranker: RerankerConfig
    llm: LLMConfig
    query_rewriter: QueryRewriterConfig
    database: DatabaseConfig
    worker: WorkerConfig
    server: ServerConfig
    ragas: RagasConfig


# ------------------------------------------------------------------ #
#  Loader — cached so the file is read exactly once per process       #
# ------------------------------------------------------------------ #

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    with open(CONFIG_PATH, "r") as f:
        raw = yaml.safe_load(f)
    return Settings(**raw)