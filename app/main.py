# app/main.py

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.routers import health, chat


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="OmniAI Backend",
        version="0.1.0",
    )

    # CORS: allow your frontend origin(s) – can tighten later
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # for now; later restrict to your Vercel domain
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, tags=["health"])
    app.include_router(chat.router, tags=["chat"])

    return app


app = create_app()
