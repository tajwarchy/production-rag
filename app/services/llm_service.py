"""
LLM abstraction layer.

LangChain makes switching providers a one-line change.
The only line you change is marked ── SWAP LINE ──  below.
Everything else — chains, prompts, RAG logic — stays identical.

Current:  Ollama (free, local, Mistral 7B)
Swap to:  OpenAI  → replace OllamaLLM(...) with ChatOpenAI(...)
Swap to:  Anthropic → replace OllamaLLM(...) with ChatAnthropic(...)
"""

from functools import lru_cache

from langchain_community.llms import Ollama as OllamaLLM
# ── SWAP LINE ── To use OpenAI instead of Ollama, comment the line above
# and uncomment one of the lines below. Everything downstream is identical.
# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger

from app.core.config import get_settings


# ------------------------------------------------------------------ #
#  Model factory — returns the configured LLM                         #
# ------------------------------------------------------------------ #

@lru_cache(maxsize=1)
def get_llm() -> BaseLanguageModel:
    cfg = get_settings().llm

    if cfg.provider == "ollama":
        # ── SWAP LINE ── This is the one line that changes per provider.
        llm = OllamaLLM(
            model=cfg.model,
            base_url=cfg.base_url,
            temperature=cfg.temperature,
            num_predict=cfg.max_tokens,
            timeout=cfg.request_timeout,
        )
        # OpenAI swap (one line):
        # llm = ChatOpenAI(model="gpt-4o-mini", temperature=cfg.temperature)
        #
        # Anthropic swap (one line):
        # llm = ChatAnthropic(model="claude-3-5-haiku-latest", temperature=cfg.temperature)

    else:
        raise ValueError(
            f"Unknown LLM provider '{cfg.provider}'. "
            "Set llm.provider in config.yaml to 'ollama', 'openai', or 'anthropic'."
        )

    logger.info("LLM initialised: provider={} model={}", cfg.provider, cfg.model)
    return llm


# ------------------------------------------------------------------ #
#  RAG answer generation                                              #
# ------------------------------------------------------------------ #

RAG_PROMPT = PromptTemplate.from_template("""
You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:""")


def generate_answer(question: str, context_chunks: list[str]) -> str:
    """
    Given a question and a list of retrieved text chunks,
    generate a grounded answer using the configured LLM.
    """
    context = "\n\n---\n\n".join(context_chunks)
    chain = RAG_PROMPT | get_llm() | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    logger.debug("LLM answer generated ({} chars)", len(answer))
    return answer.strip()