"""Configuration management for the multi-agent RAG system."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str = Field(validation_alias=("OPENAI_API_KEY", "openai_api_key"))
    openai_model_name: str = "gpt-4o-mini"
    openai_embedding_model_name: str = "text-embedding-3-large"

    # Pinecone
    pinecone_api_key: str = Field(validation_alias=("PINECONE_API_KEY", "pinecone_api_key"))
    pinecone_index_name: str = Field(validation_alias=("PINECONE_INDEX_NAME", "pinecone_index_name"))

    # Retrieval
    retrieval_k: int = 4

    model_config = SettingsConfigDict(
        env_file=".env",              # ✅ your local file is ".env"
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )