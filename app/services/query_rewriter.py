"""
Query rewriting — LangChain chain.

Ambiguous or conversational queries ("what did it say about that?")
are rewritten into self-contained, retrieval-friendly queries
("What does the document say about transformer attention mechanisms?").

If rewriting is disabled in config.yaml (query_rewriter.enabled: false),
the original query is returned unchanged.
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger

from app.core.config import get_settings
from app.services.llm_service import get_llm


REWRITE_PROMPT = PromptTemplate.from_template("""
Your task is to rewrite the user's question into a clear, specific, self-contained
search query optimised for semantic document retrieval.

Rules:
- Remove conversational filler ("can you tell me", "I was wondering", etc.)
- Make implicit references explicit
- Keep the rewritten query concise (1-2 sentences max)
- Return ONLY the rewritten query — no explanation, no preamble

Original question: {question}

Rewritten query:""")


def rewrite_query(question: str) -> str:
    """
    Rewrite an ambiguous query into a retrieval-optimised form.
    Returns the original question unchanged if rewriting is disabled.
    """
    cfg = get_settings().query_rewriter

    if not cfg.enabled:
        return question

    chain = REWRITE_PROMPT | get_llm() | StrOutputParser()
    rewritten = chain.invoke({"question": question}).strip().strip('"').strip("'")

    # Sanity check — if the LLM returns something empty or very short, fall back
    if len(rewritten) < 5:
        logger.warning("Query rewriter returned a suspiciously short result — using original")
        return question

    logger.debug("Query rewritten: '{}' → '{}'", question, rewritten)
    return rewritten