from functools import lru_cache
from typing import List

from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    env: str = "development"

    news_api_base: str = ""
    news_api_key: str = ""

    cors_origins: List[AnyHttpUrl] | List[str] = ["http://localhost:8501"]

    ollama_base: AnyHttpUrl | str = "http://localhost:11434"
    ollama_model: str = "llama3"

    api_root_path: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()
