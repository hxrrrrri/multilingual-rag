"""
LLM generation service — OpenAI-compatible API (works with vLLM, Ollama, OpenAI).
"""
from typing import List, Dict, AsyncGenerator

import httpx
from loguru import logger

from app.core.config import settings

SYSTEM_PROMPT = """You are a multilingual document assistant. Your job is to answer questions
based ONLY on the provided document excerpts. The documents may be in English, Hindi, or Malayalam.

Rules:
1. Answer only from the provided context. Do not use outside knowledge.
2. If the answer is not in the context, respond: "I could not find this information in the provided documents."
3. Cite the page number when possible (e.g., "According to page 3...").
4. Keep answers concise and factual.
5. Respond in the same language as the question."""


def build_prompt(query: str, context_chunks: List[Dict]) -> List[Dict]:
    """Build the messages array for the chat completion API."""
    context_text = "\n\n".join(
        f"[Page {c.get('page', '?')}]\n{c['text']}"
        for c in context_chunks
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {query}",
        },
    ]


async def generate_answer(
    query: str,
    context_chunks: List[Dict],
    stream: bool = False,
) -> str:
    """
    Call the LLM and return the generated answer.
    """
    messages = build_prompt(query, context_chunks)

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{settings.LLM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {settings.LLM_API_KEY}"},
            json={
                "model": settings.LLM_MODEL,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 1024,
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()
        answer = data["choices"][0]["message"]["content"]
        logger.debug(f"Generated answer ({len(answer)} chars)")
        return answer.strip()


async def stream_answer(
    query: str,
    context_chunks: List[Dict],
) -> AsyncGenerator[str, None]:
    """
    Stream the LLM answer token by token (Server-Sent Events).
    """
    messages = build_prompt(query, context_chunks)

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            f"{settings.LLM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {settings.LLM_API_KEY}"},
            json={
                "model": settings.LLM_MODEL,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 1024,
                "stream": True,
            },
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    chunk = line[6:]
                    if chunk == "[DONE]":
                        break
                    import json
                    try:
                        data = json.loads(chunk)
                        token = data["choices"][0].get("delta", {}).get("content", "")
                        if token:
                            yield token
                    except Exception:
                        pass
