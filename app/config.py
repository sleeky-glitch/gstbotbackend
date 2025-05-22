# app/config.py
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    pinecone_api_key: str
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
