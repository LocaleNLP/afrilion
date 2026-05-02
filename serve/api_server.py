#!/usr/bin/env python3
"""AfriLION OpenAI-Compatible API Server

Drop-in replacement for the OpenAI API, targeting African language quality.
Any developer who has already built an app against OpenAI can switch to
LocaleNLP by changing ONE URL and ONE API key — no SDK changes, no new
learning curve.

This is the strategic B2B entry point: developers already have the
integration built. They just need African language quality that OpenAI
cannot provide.

Endpoints:
    POST /v1/chat/completions  — OpenAI-compatible chat endpoint
    GET  /v1/models            — List available AfriLION models
    GET  /health               — Health check
    GET  /                     — API info

Usage:
    # Development
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

    # Production
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4

Client usage (drop-in OpenAI replacement):
    from openai import OpenAI
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="your-localenlp-api-key",
    )
    response = client.chat.completions.create(
        model="afrilion-1b-instruct",
        messages=[{"role": "user", "content": "Habari za asubuhi?"}]
    )

Environment variables:
    AFRILION_MODEL_PATH  — HF model ID or local path (default: AfriLION/afrilion-1b-instruct)
    AFRILION_API_KEYS    — Comma-separated valid API keys (set in production)
    AFRILION_MAX_TOKENS  — Max tokens per request (default: 2048)
    HF_TOKEN             — HuggingFace token for gated models
"""

import os
import time
import uuid
import logging
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv("AFRILION_MODEL_PATH", "AfriLION/afrilion-1b-instruct")
FALLBACK_MODEL = os.getenv("AFRILION_FALLBACK_MODEL", "HuggingFaceH4/zephyr-7b-beta")
MAX_TOKENS_LIMIT = int(os.getenv("AFRILION_MAX_TOKENS", "2048"))
HF_TOKEN = os.getenv("HF_TOKEN", None)

# API key validation — set AFRILION_API_KEYS in production
# Format: comma-separated keys, e.g. "key1,key2,key3"
_API_KEYS_RAW = os.getenv("AFRILION_API_KEYS", "")
VALID_API_KEYS: set[str] = (
    set(k.strip() for k in _API_KEYS_RAW.split(",") if k.strip())
    if _API_KEYS_RAW
    else set()  # empty = no auth required (dev mode)
)

if not VALID_API_KEYS:
    logger.warning(
        "No AFRILION_API_KEYS set — running in open mode (no authentication). "
        "Set AFRILION_API_KEYS in production."
    )

AFRILION_SYSTEM_PROMPT = """You are AfriLION, an AI assistant specialized in African languages.
You can respond in Swahili, Wolof, Hausa, Yoruba, Amharic, Zulu, Xhosa, and many other African languages.
When a user writes in an African language, respond in that language.
You are helpful, culturally aware, and knowledgeable about African contexts.
Built by LocaleNLP."""

# ---------------------------------------------------------------------------
# Lazy model loading — loaded on first request to keep startup fast
# ---------------------------------------------------------------------------
_pipeline = None


def get_pipeline():
    """Lazy-load the model pipeline."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    try:
        from transformers import pipeline as hf_pipeline, AutoTokenizer
        logger.info(f"Loading model: {MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            token=HF_TOKEN,
            use_fast=True,
        )
        _pipeline = hf_pipeline(
            "text-generation",
            model=MODEL_PATH,
            tokenizer=tokenizer,
            token=HF_TOKEN,
            device_map="auto",
            torch_dtype="auto",
        )
        logger.info(f"Model loaded: {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load {MODEL_PATH}: {e}. Falling back to HF Inference API.")
        _pipeline = "hf_inference"  # signal to use HF Inference API

    return _pipeline


# ---------------------------------------------------------------------------
# Pydantic models (OpenAI-compatible)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "afrilion-1b-instruct"
    messages: list[ChatMessage]
    max_tokens: Optional[int] = Field(default=512, le=MAX_TOKENS_LIMIT)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    stream: Optional[bool] = False
    stop: Optional[list[str]] = None
    # AfriLION extension: explicitly set target language
    # Standard OpenAI clients ignore unknown fields — so this is backward compatible
    target_language: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AfriLION API",
    description=(
        "OpenAI-compatible API for African language AI. "
        "Drop-in replacement — change one URL, keep your existing OpenAI SDK integration."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------

def verify_api_key(authorization: Optional[str]) -> None:
    """Verify Bearer token against AFRILION_API_KEYS.

    If AFRILION_API_KEYS is not set (dev mode), all requests are allowed.
    Raises HTTPException 401 on invalid key.
    """
    if not VALID_API_KEYS:
        return  # dev mode — no auth

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header. Use 'Bearer <your-api-key>'.",
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or token not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
        )


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def build_prompt(messages: list[ChatMessage], target_language: Optional[str] = None) -> str:
    """Build prompt from OpenAI-format messages using AfriLION chat template."""
    parts = []

    # Check if system message present; if not, inject AfriLION system prompt
    has_system = any(m.role == "system" for m in messages)
    if not has_system:
        system_content = AFRILION_SYSTEM_PROMPT
        if target_language and target_language.lower() != "auto":
            system_content += f"\nPlease respond in {target_language}."
        parts.append(f"<|system|>\n{system_content}<|end|>")

    for msg in messages:
        if msg.role == "system":
            content = msg.content
            if target_language and target_language.lower() != "auto":
                content += f"\nPlease respond in {target_language}."
            parts.append(f"<|system|>\n{content}<|end|>")
        elif msg.role == "user":
            parts.append(f"<|user|>\n{msg.content}<|end|>")
        elif msg.role == "assistant":
            parts.append(f"<|assistant|>\n{msg.content}<|end|>")

    parts.append("<|assistant|>")
    return "\n".join(parts)


async def generate_response(
    request: ChatCompletionRequest,
) -> tuple[str, int, int]:
    """Generate response. Returns (text, prompt_tokens, completion_tokens)."""
    pipe = get_pipeline()
    prompt = build_prompt(request.messages, request.target_language)

    if pipe == "hf_inference":
        # Fallback: use HuggingFace Inference API
        from huggingface_hub import InferenceClient
        client = InferenceClient(model=FALLBACK_MODEL, token=HF_TOKEN)
        messages_hf = [
            {"role": m.role, "content": m.content}
            for m in request.messages
        ]
        result = client.chat_completion(
            messages_hf,
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature or 0.7,
            top_p=request.top_p or 0.95,
        )
        text = result.choices[0].message.content
        prompt_tokens = len(prompt.split())
        completion_tokens = len(text.split())
        return text, prompt_tokens, completion_tokens
    else:
        # Local model inference
        outputs = pipe(
            prompt,
            max_new_tokens=request.max_tokens or 512,
            temperature=request.temperature or 0.7,
            top_p=request.top_p or 0.95,
            do_sample=True,
            pad_token_id=pipe.tokenizer.eos_token_id,
            eos_token_id=pipe.tokenizer.eos_token_id,
        )
        generated = outputs[0]["generated_text"]
        # Extract only the assistant response (after the last <|assistant|>)
        if "<|assistant|>" in generated:
            text = generated.split("<|assistant|>")[-1].strip()
            # Remove trailing end tokens
            for stop_token in ["<|end|>", "<|user|>", "<|system|>"]:
                if stop_token in text:
                    text = text[:text.index(stop_token)].strip()
        else:
            text = generated[len(prompt):].strip()

        prompt_tokens = len(pipe.tokenizer.encode(prompt))
        completion_tokens = len(pipe.tokenizer.encode(text))
        return text, prompt_tokens, completion_tokens


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "name": "AfriLION API",
        "version": "0.1.0",
        "description": "OpenAI-compatible API for African language AI by LocaleNLP",
        "docs": "/docs",
        "models": "/v1/models",
        "chat": "/v1/chat/completions",
        "supported_languages": [
            "Swahili (sw)", "Wolof (wo)", "Hausa (ha)", "Yoruba (yo)",
            "Amharic (am)", "Zulu (zu)", "Xhosa (xh)", "Igbo (ig)",
            "Somali (so)", "Tigrinya (ti)", "Shona (sn)", "Luganda (lg)",
            "Twi (tw)", "French-West-Africa (fr-WA)", "+ more",
        ],
        "github": "https://github.com/LocaleNLP/afrilion",
        "huggingface": "https://huggingface.co/AfriLION",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_PATH, "timestamp": int(time.time())}


@app.get("/v1/models")
async def list_models(authorization: Optional[str] = Header(None)):
    verify_api_key(authorization)
    return {
        "object": "list",
        "data": [
            {
                "id": "afrilion-1b-instruct",
                "object": "model",
                "created": 1746144000,  # May 2, 2026
                "owned_by": "LocaleNLP",
                "description": "AfriLION 1B instruction-following model for African languages",
                "context_length": 2048,
                "languages": [
                    "sw", "wo", "ha", "yo", "am", "zu", "xh",
                    "ig", "so", "ti", "sn", "lg", "tw",
                ],
            },
            {
                "id": "afrilion-1b-instruct-gptq",
                "object": "model",
                "created": 1746144000,
                "owned_by": "LocaleNLP",
                "description": "AfriLION 1B GPTQ 4-bit quantized — faster inference, same quality",
                "context_length": 2048,
                "languages": [
                    "sw", "wo", "ha", "yo", "am", "zu", "xh",
                    "ig", "so", "ti", "sn", "lg", "tw",
                ],
            },
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None),
):
    """OpenAI-compatible chat completions endpoint.

    This is a drop-in replacement for the OpenAI API.
    Switch your base_url from https://api.openai.com/v1 to
    https://api.localenlp.com/v1 (or your self-hosted URL).

    STRATEGIC NOTE:
    By maintaining full OpenAI API compatibility, any B2B customer who
    has already integrated OpenAI can switch to LocaleNLP with zero code
    changes on their end — just a URL and API key swap. This is the
    lowest-friction enterprise sales motion possible.
    """
    verify_api_key(authorization)

    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    request_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())

    if request.stream:
        # Streaming response (SSE)
        async def stream_generator() -> AsyncGenerator[str, None]:
            try:
                # For streaming, we generate the full response then chunk it
                # TODO: Replace with true token streaming when using vLLM/llama.cpp server
                text, prompt_tokens, completion_tokens = await generate_response(request)
                words = text.split()

                for i, word in enumerate(words):
                    chunk_text = word + (" " if i < len(words) - 1 else "")
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": chunk_text},
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    # Small delay for realistic streaming feel
                    import asyncio
                    await asyncio.sleep(0.01)

                # Final chunk with finish_reason
                final_chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }],
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                error_chunk = {"error": {"message": str(e), "type": "server_error"}}
                yield f"data: {json.dumps(error_chunk)}\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    else:
        # Non-streaming response
        try:
            text, prompt_tokens, completion_tokens = await generate_response(request)
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

        return ChatCompletionResponse(
            id=request_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )


# ---------------------------------------------------------------------------
# Startup events
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    logger.info("AfriLION API server starting...")
    logger.info(f"  Model: {MODEL_PATH}")
    logger.info(f"  Auth: {'enabled' if VALID_API_KEYS else 'disabled (dev mode)'}")
    logger.info(f"  Max tokens: {MAX_TOKENS_LIMIT}")
    logger.info("  Docs: http://localhost:8000/docs")
    # Pre-warm the model in background
    import asyncio
    asyncio.create_task(asyncio.to_thread(get_pipeline))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=os.getenv("DEBUG", "").lower() in ("1", "true", "yes"),
        log_level="info",
    )
