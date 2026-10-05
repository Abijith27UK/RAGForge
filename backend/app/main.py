"""RAGForge FastAPI application entrypoint."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging("DEBUG" if settings.debug else "INFO")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    logger.info("RAGForge starting; data dir: %s", settings.data_dir)
    yield


app = FastAPI(
    title="RAGForge API",
    version="0.1.0",
    description="Automated Domain-Specific RAG Knowledge Base Builder",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # The configured frontend origin is honoured rather than hardcoded, so a
    # developer running the UI on a different port does not get a silent
    # "Failed to fetch" with no server-side hint. The two loopback dev origins
    # stay allowed because FRONTEND_URL usually names only one of them.
    allow_origins=sorted(
        {
            get_settings().frontend_url,
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        }
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
from app.api.routes_knowledge_bases import router as kb_router
from app.api.routes_domain import router as domain_router
from app.api.routes_sources import router as sources_router
from app.api.routes_build import router as build_router
from app.api.routes_documents import router as documents_router
from app.api.routes_corpus import router as corpus_router
from app.api.routes_retrieval import router as retrieval_router
from app.api.routes_answer import router as answer_router
from app.api.routes_chat import router as chat_router
from app.api.routes_answer_eval import router as answer_eval_router
from app.api.routes_answer_eval import global_router as answer_eval_global_router
from app.api.routes_benchmark import router as benchmark_router
from app.api.routes_experiments import router as experiments_router
from app.api.routes_system import router as system_router

app.include_router(kb_router)
app.include_router(domain_router)
app.include_router(sources_router)
app.include_router(build_router)
app.include_router(documents_router)
app.include_router(corpus_router)
app.include_router(retrieval_router)
app.include_router(answer_router)
app.include_router(chat_router)
app.include_router(answer_eval_router)
app.include_router(answer_eval_global_router)
app.include_router(benchmark_router)
app.include_router(experiments_router)
app.include_router(system_router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Never leak stack traces to the frontend; log them instead."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/")
def root():
    return {"app": "RAGForge API", "docs": "/docs", "health": "/api/system/health"}
