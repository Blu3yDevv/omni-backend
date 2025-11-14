# app/routers/health.py

from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings

router = APIRouter()


@router.get("/health")
async def health_check():
    settings = get_settings()
    return {
        "status": "ok",
        "env": settings.ENV,
        "debug": settings.DEBUG,
    }
