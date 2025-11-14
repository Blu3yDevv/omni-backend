# app/core/config.py

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings



class Settings(BaseSettings):
    # General
    ENV: str = os.getenv("ENV", "local")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # LLM Space
    LLM_SPACE_ID: str = os.getenv("LLM_SPACE_ID", "Blu3yDevv/omni-nano-inference")
    HF_API_TOKEN: Optional[str] = (
        os.getenv("HF_API_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY")
    )

    # RAG / Qdrant (for later when we re-enable RAG)
    QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")

    # Supabase (for later)
    SUPABASE_URL: Optional[str] = os.getenv("SUPABASE_URL")
    SUPABASE_ANON_KEY: Optional[str] = os.getenv("SUPABASE_ANON_KEY")
    SUPABASE_SERVICE_KEY: Optional[str] = os.getenv("SUPABASE_SERVICE_KEY")

    class Config:
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()
