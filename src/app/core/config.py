"""Configuration management for the multi-agent RAG system."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str
    openai_model_name: str = "gpt-4o-mini"
    openai_embedding_model_name: str = "text-embedding-3-large"

    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str

    # Retrieval
    retrieval_k: int = 4

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
